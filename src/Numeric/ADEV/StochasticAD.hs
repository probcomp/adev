module Numeric.ADEV.StochasticAD (PruningProgram(..), stochastic_derivative) where

import Control.Applicative -- Otherwise you can't do the Applicative instance.
import Control.Monad (liftM, ap)
import Control.Monad.Bayes.Class (MonadDistribution, bernoulli)

-- | Type used to represent translated "Stochastic AD" programs.
-- The idea is that a value of type PruningProgram m a pairs together:
-- * a "stacked" Stochastic Program (a la Arya et al. 2022) X(p) 
--   over *traces* of type [Double], recording all the user's random
--   choices.
-- * a function that takes such a trace and outputs a value of type a.
-- The trace of the program stores Doubles, but these may be the results
-- of discrete choices; for example, drawing `False` from a Bernoulli
-- will store `0.0` in the trace, and drawing `True` will store `1.0`.
-- A PruningProgram knows how to sample its stochastic derivative (again, 
-- in the sense of Arya et al. 2022). In particular, it knows how to 
-- generate both a primary trace *and* an alternative trace, as well as a weight,
-- where the alternative trace differs at most in one random choice from the 
-- primary trace, and the weight is calculated as in Arya et al. 2022.
-- The implementation is a bit weird:
-- * The `runPruned` method returns three things. Intuitively, they should be
--   the primary trace, the alternative trace, and the weight. However, the first
--   return value is not the primary trace, but rather something lower-level: we 
--   call it `u`, and it corresponds roughly to $\Omega$ in Arya et al. 2022.
--   It records the underlying randomness used to generate a primary trace, and 
--   can be converted into a real primary trace using the `getTrace` method.
--   For example, if the program generates two `bernoulli 0.5` variables, then
--   the underlying trace might be `[0.1, 0.7]`. Applying `getTrace` to this
--   underlying trace would yield the primary trace `[0, 1]`, representing False
--   and True.
-- * The `getRetval` method converts a trace into a value of type a. For example,
--   for a program that samples two Bernoullis and then returns `b1 && b2`, the
--   return value for the trace `[0.0, 1.0]` would be `False`.
-- * Both the `getTrace` and `getRetval` methods can accept input lists that contain
--   excess randomness, beyond what the program could sample in one run. In this case,
--   any excess randomness is returned as a second return value.
data PruningProgram m a = PruningProgram { 
  runPruned :: m ([Double], [Double], Double),
  getTrace  :: [Double] -> m ([Double], [Double]),
  getRetval :: [Double] -> (a, [Double]) }

stochastic_derivative :: MonadDistribution m => PruningProgram m a -> m (a, a, Double)
stochastic_derivative prog = do
  (u, t', w)  <- runPruned prog
  (t, _)      <- getTrace prog u
  let (v, _)  =  getRetval prog t
  let (v', _) =  getRetval prog t'
  return (v, v', w)

instance MonadDistribution m => Functor (PruningProgram m) where
  fmap = liftM

instance MonadDistribution m => Applicative (PruningProgram m) where
  pure  = return
  (<*>) = ap

instance MonadDistribution m => Monad (PruningProgram m) where
  -- | The `return` method is used to construct a program that does not sample any
  -- random variables, and returns a specified value. Its traces (primary and alternate)
  -- are empty and its weight is 0.0. The `getRetval` function returns `x` no matter what.
  return x = PruningProgram (return ([], [], 0)) (\u -> return ([], u)) (\t -> (x, t))
  
  -- | The `>>=` method is used to sequence probabilistic computations.
  -- It first samples a trace from `mu`, then uses `getRetval` to extract a value `v1`.
  -- It then samples the final part of the trace from `k v1`.
  -- The key complexity here is in how the alternative trace is computed.
  -- We use the "aggressive pruning" strategy of Arya et al.: a weight is computed
  -- for each program in the sequence, and then a Bernoulli is sampled with that weight
  -- to determine which of the two halves contains the alternative that will be tracked.
  -- The other alternative is discarded, and the weight becomes the sum of the weights.
  mu >>= k = PruningProgram run_ trace_ ret_
    where
      run_ = do
        (u1, t1', w1) <- runPruned mu
        (t1, _)       <- getTrace mu u1
        let (v1, _)   =  getRetval mu t1
        (u2, t2', w2) <- runPruned (k v1)
        let weight    =  w1 + w2
        b             <- bernoulli (w1 / weight)
        if b then do
          let (v1', _) =  getRetval mu t1'
          (t2,  _)     <- getTrace (k v1') u2
          return (u1 ++ u2, t1' ++ t2, weight)
        else
          return (u1 ++ u2, t1 ++ t2', weight)

      trace_ u = do
        (t1, u2)    <- getTrace mu u
        let (v1, _) =  getRetval mu t1
        (t2, u3)    <- getTrace (k v1) u2
        return (t1 ++ t2, u3)

      ret_ t = 
        let (v1, t2) = getRetval mu t in
        let (v2, t3) = getRetval (k v1) t2 in
        (v2, t3)