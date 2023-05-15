{-# LANGUAGE FlexibleInstances, FunctionalDependencies #-}

module Numeric.ADEV.DiffOptimized (
  ADEV(..), diff
) where

import Numeric.ADEV.Class
import Numeric.ADEV.Interp()
import Control.Monad.Bayes.Class (
  MonadDistribution, 
  uniform, 
  uniformD,
  logCategorical,
  poisson,
  bernoulli,
  normal)
import Control.Monad.Bayes.Sampler.Lazy (Sampler)
import Control.Monad.Bayes.Sampler.Lazy.Coupled (coupled)
import Control.Monad.Cont (ContT(..))
import Numeric.AD.Internal.Forward.Double (
  ForwardDouble, 
  bundle, 
  primal, 
  tangent)
import Control.Monad (replicateM, mapM)
import Numeric.Log (Log(..))
import qualified Numeric.Log as Log (sum)
import Data.List (zipWith4)
import qualified Data.Vector as V
import Numeric.ADEV.StochasticAD (stochastic_derivative, PruningProgram(..))

split :: ForwardDouble -> (Double, Double)
split dx = (primal dx, tangent dx)

-- | ADEV translation of an ADEV program.
-- Implements the built-in derivative for every ADEV primitive.
--   * Reals are interpreted as ForwardDoubles, pairs of Doubles.
--   * Underlying randomness is provided by a monad @m@ satisfying the 
--     monad-bayes @MonadSample@ interface.
--   * ADEV probabilistic programs are represented by the monad
--     @ContT ForwardDouble m@: they know how to transform estimators of
--     losses and loss derivatives into estimators of *expected* losses and
--     loss derivatives, where the expectation is taken over the probabilistic
--     program in question.

call dloss b x = do
  l <- dloss x
  l b

instance ADEV (ContT (Bool -> Sampler ForwardDouble)) Sampler ForwardDouble PruningProgram where
  sample = ContT $ \dloss -> return $ \wants_grad -> do
    u <- uniform 0 1
    call dloss wants_grad (bundle u 0)
  
  flip_enum dp = ContT $ \dloss -> return $ \wants_grad -> do
    (dl1, dl2) <- coupled (call dloss wants_grad True) (call dloss wants_grad False)
    return (bundle (primal $ dp * dl1 + (1 - dp) * dl2) 0)

  flip_reinforce dp = ContT $ \dloss -> return $ \wants_grad -> do
    b           <- bernoulli (primal dp)
    (l, l')     <- fmap split (call dloss wants_grad b)
    if wants_grad then
      let logpdf' = tangent (log $ if b then dp else 1 - dp) in
      return (bundle l (l' + l * logpdf'))
    else
      return (bundle l 0)

  normal_reparam dmu dsig = do
    deps <- stdnorm
    return $ (deps * dsig) + dmu
    where 
      stdnorm = ContT $ \dloss -> return $ \wants_grad -> do
        eps <- normal 0 1
        call dloss wants_grad (bundle eps 0)

  normal_reinforce dmu dsig = ContT $ \dloss -> return $ \wants_grad -> do
    x           <- normal (primal dmu) (primal dsig)
    let dx      =  bundle x 0
    (l, l')     <- fmap split (call dloss wants_grad dx)
    if wants_grad then
      let logpdf' =  tangent $ (-1 * log dsig) - 0.5 * ((dx - dmu) / dsig)^2 in
      return (bundle l (l' + l * logpdf'))
    else
      return (bundle l 0)
    
  add_cost dcost = ContT $ \dloss -> return $ \wants_grad -> do
    dl <- call dloss wants_grad ()
    return (dl + dcost)
   
  expect prog = do
    dloss <- runContT prog (\dl -> return $ \wants_grad -> return (bundle (primal dl) (if wants_grad then tangent dl else 0)))
    dloss True
  
  plus_ estimate_da estimate_db = do -- different from paper's estimator
    da <- estimate_da
    db <- estimate_db
    return (da + db)
  
  times_ estimate_da estimate_db = do
    da <- estimate_da
    db <- estimate_db
    return (da * db)
    
  exp_ estimate_dx = do
    (x, x') <- (fmap split estimate_dx)
    s <- exp_ (fmap primal estimate_dx)
    return (bundle x (s * x'))
  
  minibatch_ n m estimate_df = do
    indices <- replicateM m (uniformD [1..n])
    dfs <- mapM (\i -> estimate_df i) indices
    return $ (sum dfs) * (fromIntegral n / fromIntegral m)
  
  exact_ = return

  baseline dp db  = do
    get_dl <- runContT dp (\dx -> return $ \wants_grad -> return (bundle (primal dx - primal db) (if wants_grad then tangent dx - tangent db else 0)))
    dl <- get_dl True
    return (dl + db)

  reinforce (D dsamp dpdf) = ContT $ \dloss -> return $ \wants_grad -> do
    x <- dsamp
    (l, l') <- fmap split (call dloss wants_grad x)
    if wants_grad then
      let logpdf' = tangent $ ln (dpdf x) in
      return (bundle l (l' + l * logpdf'))
    else
      return (bundle l 0)

  leave_one_out m (D dsamp dpdf) = ContT $ \dloss -> return $ \wants_grad -> do
    xs <- replicateM m dsamp
    dlosses <- mapM (call dloss wants_grad) xs
    if not wants_grad then
      return (bundle (sum (map primal dlosses) / fromIntegral m) 0)
    else do
      let (ls, l's) = unzip (map split dlosses)
      -- For each l, average the other ls to get a baseline
      let bs = map (\i -> (sum (take i ls) + sum (drop (i + 1) ls)) / (fromIntegral (m - 1))) [0..m-1]
      let logpdfs = map (tangent . ln . dpdf) xs
      return $ bundle (sum ls / fromIntegral m) (sum (zipWith4 (\l l' b lpdf -> l' + (l - b) * lpdf) ls l's bs logpdfs) / fromIntegral m)

  implicit_reparam (C samp pdf dcdf) = ContT $ \dloss -> return $ \wants_grad -> do
    x <- samp
    if not wants_grad then
      call dloss False (bundle x 0)
    else do
      let f' = tangent $ (exp . ln . dcdf) (bundle x 0)
      let p  = (exp . ln . pdf) x
      call dloss True (bundle x (-f' / p))

  poisson_weak drate = ContT $ \dloss -> return $ \wants_grad -> do
    let (rate, rate') = split (exp (ln drate))
    x <- poisson rate
    if not wants_grad then do
      -- Total hack: we split the following into two lines,
      -- so that it uses RNG the exact same way as the else
      -- branch, to improve coupling.
      y <- call dloss False x
      return y
    else do
      (y_neg, y_pos) <- coupled (call dloss True x) (call dloss False (x + 1))
      let grad = primal y_pos - primal y_neg
      return (bundle (primal y_neg) (grad * rate'))

  reparam_reject (D s spdf) h (D p ppdf) (D q qpdf) m = ContT $ \dloss -> 
    runContT (reinforce dpi) (dloss . h)
    where
    pi = do
      eps <- s 
      let x = h eps
      let w = exp ((primal (ln (ppdf x))) - (primal (ln (qpdf x))))
      u <- uniform 0 1
      if u < w then return eps else pi
    dpi_density deps = spdf deps * ppdf (h deps) / qpdf (h deps)
    dpi = D pi dpi_density
  
  smc dp (D q0samp q0dens) dq df n k = do
    particles <- iterateM step init n
    values <- mapM (\(v, w) -> do
      (f, f') <- fmap split (df v)
      let logpdf' = tangent $ ln (dp v)
      return (bundle f (exp (ln w) * (f' + f * logpdf')))) particles
    return $ sum values / fromIntegral k
    where
    iterateM k m n = if n == 0 then m else do
      x <- m
      iterateM k (k x) (n - 1)
    pp = Exp . primal . ln . dp
    qq0 = Exp . primal . ln . q0dens
    init = replicateM k (do
      x <- q0samp
      return ([x], pp [x] / qq0 x))
    resample particles = do
      let weights = map snd particles
      let total_weight = Log.sum weights
      let normed_weights = map (\w -> w / total_weight) weights
      indices <- replicateM k (logCategorical (V.fromList normed_weights))
      let new_weights = replicate k (total_weight / fromIntegral k)
      return $ zip (map (\i -> fst (particles !! i)) indices) new_weights
    propagate particle = do
      let (v, w) = particle
      let (D qs qd) = dq (head v)
      let qqd = Exp . primal . ln . qd
      v' <- qs
      return (v':v, w * (pp (v':v) / pp v) / qqd v')
    step particles = do 
      particles <- resample particles
      mapM propagate particles

  flip_pruned dp = PruningProgram run_ trace_ ret_
    where
      (p, p') = split dp
      run_    = do
        u <- uniform 0 1
        let b = u > (1 - p)
        if dp > 0 then
          return ([u], [1.0], if b then 0 else p'/(1-p))
        else
          return ([u], [0.0], if b then -p'/p else 0)
      trace_ [] = do 
        u <- uniform 0 1
        return ([if u > (1 - p) then 1 else 0], [])
      trace_ (u:us) = return ([if u > (1 - p) then 1 else 0], us)
      ret_   (t:ts) = (t == 1.0, ts) -- TODO: use t > 0.5 instead, for floating-point safety?
  
  normal_pruned mu sig = PruningProgram run_ trace_ ret_
    where
      run_    = do
        u <- normal 0 1
        return ([u], [u], 0)
      trace_ [] = do 
        u <- normal 0 1
        return ([u], [])
      trace_ (u:us) = return ([u], us)
      ret_   (t:ts) = ((bundle t 0) * sig + mu, ts)

  expect_pruned = expect . run_pruned
  
  run_pruned pruning_prog = ContT $ \dloss -> return $ \wants_grad -> do
    (val, val', weight) <- stochastic_derivative pruning_prog
    if not wants_grad || weight == 0.0 then do
      l <- call dloss wants_grad val
      return l
    else do
      ((l1, l1'), l2) <- coupled (fmap split (call dloss True val)) (fmap primal (call dloss False val'))
      return (bundle l1 (l1' + (l2 - l1) * weight))

  flip_weak dp = ContT $ \dloss -> return $ \wants_grad -> do
    b <- bernoulli (primal dp)
    if wants_grad then do
      (l1, l2) <- coupled (call dloss True b) (call dloss False (not b))
      let (ltrue, lfalse) = if b then (l1, l2) else (l2, l1)
      let loss_est = (primal dp * primal ltrue) + ((1 - primal dp) * primal lfalse)
      let grad_est = tangent l1 + (primal ltrue - primal lfalse) * (tangent dp)
      return (bundle loss_est grad_est)
    else do
      dl <- call dloss False b
      return (bundle (primal dl) 0)

diff :: MonadDistribution m => (ForwardDouble -> m ForwardDouble) -> Double -> m Double
diff f x = do
  df <- f (bundle x 1)
  return (tangent df)