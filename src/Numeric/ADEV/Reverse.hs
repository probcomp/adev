{-# LANGUAGE FlexibleInstances, TypeFamilies, FlexibleContexts, MultiParamTypeClasses #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}

module Numeric.ADEV.Reverse (
  ADEV(..)
) where

import Numeric.ADEV.Class ( ADEV(..) )
import Numeric.ADEV.Interp()
import Control.Monad.Bayes.Class (
  uniform,
  poisson,
  bernoulli,
  normal)
import Control.Monad.Bayes.Sampler.Lazy.Coupled (coupled)
import Control.Monad.Cont (ContT(..))

import Numeric.Log (Log(..))
import Numeric.AD.Internal.Reverse ( primal, Reverse, Tape )
import Numeric.AD.Mode ( Mode(auto) )
import Numeric.AD.Jacobian ( Jacobian(lift2) )
import Data.Reflection ( Reifies )
import Control.Monad.Bayes.Sampler.Lazy ( SamplerT )

instance (Reifies s Tape) => ADEV (ContT (Reverse s Double)) SamplerT (Reverse s Double) where

  unif = ContT $ \dloss -> do
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
