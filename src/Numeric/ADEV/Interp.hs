{-# LANGUAGE FlexibleInstances, ImportQualifiedPost, FunctionalDependencies #-}

module Numeric.ADEV.Interp where
  
import Numeric.ADEV.Class
import Control.Monad.Bayes.Class (
  MonadDistribution, 
  uniform, 
  uniformD,
  logCategorical,
  score, 
  bernoulli, 
  poisson,
  normal)
import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.Writer.Lazy (WriterT(..), tell)
import Data.Monoid (Sum(..))
import Control.Monad (replicateM, mapM)
import Data.Vector qualified as V
import Numeric.Log (Log(..))
import qualified Numeric.Log as Log (sum)

-- | Standard, non-AD semantics of an ADEV program.
--   * Reals are represented as Doubles.
--   * Randomness comes from an underlying measure monad @m@ satisfying
--     the monad-bayes @MonadInfer@ interface.
--   * The ADEV probability monad is interpreted as @WriterT (Sum Double) m@,
--     i.e. a probabilistic computation that accumulates an additive loss.
instance MonadDistribution m => ADEV (WriterT (Sum Double)) m Double where
  sample           = uniform 0 1
  flip_enum        = bernoulli
  flip_reinforce   = bernoulli
  normal_reparam   = normal
  normal_reinforce = normal
  add_cost w       = tell (Sum w)
  expect f         = do {(x, w) <- runWriterT f; return (x + getSum w)}
  exact_           = return
  plus_ esta estb  = do
    a <- esta
    b <- estb
    return (a + b)
  times_ esta estb = do
    a <- esta
    b <- estb
    return (a * b)
  exp_ estx        = do
    n  <- poisson rate
    xs <- replicateM n estx
    return $ exp rate * product (map (\x -> x / rate) xs)
    where rate = 2
  minibatch_ n m f = do
    indices <- replicateM m (uniformD [1..n])
    vals    <- mapM f indices
    return $ (fromIntegral n / fromIntegral m) * (sum vals)
  baseline p b     = expect p
  reinforce (D sampler density) = lift sampler
  leave_one_out n (D sampler density) = lift sampler
  smc p (D q0samp q0dens) q f n k = do
    particles <- iterateM step init n
    values <- mapM (\(v, w) -> do
      x <- f v
      return (x * exp (ln w))) particles
    return $ sum values / fromIntegral k
    where
    iterateM k m n = if n == 0 then m else do
      x <- m
      iterateM k (k x) (n - 1)
    init = replicateM k (do
      x <- q0samp
      return ([x], p [x] / q0dens x))
    resample particles = do
      let weights = map snd particles
      let total_weight = Log.sum weights
      let normed_weights = map (\w -> w / total_weight) weights
      indices <- replicateM k (logCategorical (V.fromList normed_weights))
      let new_weights = replicate k (total_weight / fromIntegral k)
      return $ zip (map (\i -> fst (particles !! i)) indices) new_weights
    propagate particle = do
      let (v, w) = particle
      let (D qs qd) = q (head v)
      v' <- qs
      return (v':v, (p (v':v) / p v) / qd v')
    step particles = do 
      particles <- resample particles
      mapM propagate particles
  importance (D samp _) _ = lift samp
  implicit_reparam (C samp pdf cdf) = lift samp
  poisson_weak (Exp rate) = poisson (exp rate)
  reparam_reject s h (D p ppdf) (D q qpdf) m = do
    x <- lift q
    let w = ppdf x / (m * qpdf x)
    u <- uniform 0 1
    if log u < ln w then do
      return x
    else
      reparam_reject s h (D p ppdf) (D q qpdf) m