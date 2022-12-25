{-# LANGUAGE MultiParamTypeClasses, FunctionalDependencies #-}

module Numeric.ADEV.Class (
  D(..), C(..), ADEV(..)) where

import Numeric.Log

-- | Type of density-carrying distributions.
data D m r a = D (m a) (a -> Log r)

-- | Type of CDF-carrying distributions over the reals, 
-- for implicit reparameterization.
data C m r = C (m Double) (Double -> Log Double) (r -> Log r)

-- ---------------------------------------------------------------------------
-- | A typeclass for ADEV programs, parameterized by:
-- 
--   * @r@ - the type used to represent real numbers
--   * @m@ - the monad used to encode randomness (so that @m r@ is the type of
--           unbiasedly estimated real numbers)
--   * @p@ - the type used for monadic probabilistic programming (so 
--           that @p m a@ is a probabilistic program returning @a@)
--   * @s@ - the type used for monadic probabilistic programming with
--           Stochastic AD (so that @s m a@ is a probabilistic program
--           returning @a@ handled by Arya et al. (2022)'s AD scheme.)
class (RealFrac r, Monad (p m), Monad m, Monad (s m)) => ADEV p m r s | p -> r, r -> p, r -> s, s -> r where
  -- | Sample a random uniform value between 0 and 1.
  sample           :: p m r
  -- | Add a real value into a running cost accumulator.
  -- When a @p m r@ is passed to @expect@, the result is
  -- an estimator of the expected cost *plus* the expected
  -- return value.
  add_cost         :: r -> p m ()
  -- | Flip a coin with a specified probability of heads.
  -- Uses enumeration (costly but low-variance) to estimate
  -- gradients.
  flip_enum        :: r -> p m Bool
  -- | Flip a coin with a specified probability of heads.
  -- Uses the REINFORCE estimator (cheaper but higher-variance) 
  -- for gradients.
  flip_reinforce   :: r -> p m Bool
  -- | Generate from a normal distribution. Uses the REPARAM gradient estimator.
  normal_reparam   :: r -> r -> p m r
  -- | Generate from a normal distribution. Uses the REINFORCE gradient estimator.
  normal_reinforce :: r -> r -> p m r
  -- | Estimate the expectation of a probabilistic computation.
  expect           :: p m r -> m r
  -- | Combinator DSL for estimators
  plus_            :: m r -> m r -> m r
  times_           :: m r -> m r -> m r
  exp_             :: m r -> m r
  minibatch_       :: Int -> Int -> (Int -> m r) -> m r
  exact_           :: r -> m r
  -- | Baselines for controlling variance
  baseline         :: p m r -> r -> m r
  -- | Automatic construction of new REINFORCE estimators
  reinforce        :: D m r a -> p m a
  -- | Storchastic leave_one_out estimator
  leave_one_out    :: Int -> D m r a -> p m a
  -- | Differentiable particle filter, accepting:
  --   * @p@: a density function for the target measure.
  --   * @q0@: an initial proposal for the particle filter.
  --   * @q@: a transition proposal for the particle filter.
  --   * @f@: an unbiased estimator of an integrand to estimate
  --   * @n@: the number of SMC steps to run
  --   * @k@: the number of particles to use
  -- Returns an SMC estimator of the integral
  smc :: ([a] -> Log r) -> D m r a -> (a -> D m r a) -> ([a] -> m r) -> Int -> Int -> m r
  -- | Importance sampling gradient estimator
  importance       :: D m r a -> D m r a -> p m a
  -- | Implicit reparameterization for real-valued distributions 
  -- differentiable with CDFs (e.g., mixtures of Gaussians)
  implicit_reparam :: C m r -> p m r
  -- | Sample from a Poisson distribution, using a measure-valued derivative.
  poisson_weak     :: Log r -> p m Int
  -- | Gradients through rejection sampling for density-carrying distributions.
  reparam_reject   :: D m r a -> (a -> b) -> (D m r b) -> (D m r b) -> Log r -> p m b
  -- | Stochastic AD
  flip_pruned      :: r -> s m Bool
  normal_pruned    :: r -> r -> s m r
  expect_pruned    :: s m r -> m r -- is now just expect . run_pruned
  run_pruned       :: s m a -> p m a -- New: run a stochastic AD program inside an ADEV probabilistic program
