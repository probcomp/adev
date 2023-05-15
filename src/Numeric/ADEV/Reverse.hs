{-# LANGUAGE FlexibleInstances, TypeFamilies, FlexibleContexts, FunctionalDependencies #-}

module Numeric.ADEV.Reverse (
  ADEV(..)
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

import Control.Monad (replicateM, mapM)
import Numeric.Log (Log(..))
import qualified Numeric.Log as Log (sum)
import Data.List (zipWith4)
import qualified Data.Vector as V
import Numeric.ADEV.StochasticAD (stochastic_derivative, PruningProgram(..))
import Numeric.AD.Internal.Reverse
import Numeric.AD.Mode
import Numeric.AD.Jacobian
import Numeric.AD.Mode.Reverse
import Data.Reflection
import Control.Monad.Bayes.Sampler.Lazy

instance (Reifies s Tape) => ADEV (ContT (Reverse s Double)) Sampler (Reverse s Double) PruningProgram where

  sample = ContT $ \dloss -> do
    u <- uniform 0 1
    dloss (auto u)

  flip_reinforce p = ContT $ \dloss -> do
    b <- bernoulli (primal p)
    l <- dloss b
    let lpdf = log $ if b then p else 1 - p
    let deriv lpdf' l' = (l', 1)
    return $ lift2 (\_ l' -> l') deriv lpdf l

  flip_enum p = ContT $ \dloss -> do
    (l_true, l_false)  <- coupled (dloss True) (dloss False)
    return (p * l_true + (1 - p) * l_false)

  add_cost c = ContT $ \dloss -> do
    l <- dloss ()
    return (l + c)

  normal_reparam mu sigma = ContT $ \dloss -> do
    eps <- normal 0 1
    dloss (mu + sigma * (auto eps))

  normal_reinforce mu sigma = ContT $ \dloss -> do
    x <- normal (primal mu) (primal sigma)
    l <- dloss (auto x)
    let lpdf = (-1 * log sigma) - 0.5 * ((auto x - mu) / sigma)^2
    let deriv lpdf' l' = (l', 1)
    return $ lift2 (\_ l' -> l') deriv lpdf l

  poisson_weak (Exp log_lambda) = ContT $ \dloss -> do
    let lambda = exp log_lambda
    k_neg <- poisson (primal lambda)
    let k_pos = k_neg + 1
    (l_neg, l_pos) <- coupled (dloss k_neg) (dloss k_pos)
    let l_pos' = primal l_pos
    return $ lift2 (\lambda' l_neg' -> l_neg') (\lambda' l_neg' -> (realToFrac l_pos' - l_neg', 1)) lambda l_neg

  expect prog = runContT prog return
