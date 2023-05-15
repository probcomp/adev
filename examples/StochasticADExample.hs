module Main (main) where

import Numeric.ADEV.Class
import Numeric.ADEV.DiffOptimized (diff)
import Numeric.ADEV.Interp ()
import Numeric.AD.Internal.Forward.Double (ForwardDouble, bundle)
import Control.Monad.Bayes.Class (MonadDistribution)
import Control.Monad.Bayes.Sampler.Lazy (sampler)
import Control.Monad (replicateM)
import Numeric.ADEV.StochasticAD (stochastic_derivative)

-- Examples of ADEV's implementation of Stochastic AD (Arya et al. 2022).
-- Currently, only the "Aggressive Pruning" strategy is supported, and
-- a Stochastic Program (in the sense of Arya et al. 2022) cannot be 
-- embedded within an ADEV probabilistic program; rather, the user can
-- apply `expect_pruned` to yield a differentiable estimator of the program's expectation.
-- The user can further apply exp_, times_, plus_, etc. to the estimator.

-- General control flow is supported. However,
-- as in the rest of ADEV, using comparison operators like < 
-- with real numbers that depend directly on the parameter 
-- being differentiated is unsound. This includes the results of 
-- calls to `normal_pruned`, which is essentially `normal_reparam` but
-- for use within Stochastic Programs.

-- Helper: estimate derivative with N samples
diffN n l x = do
  xs <- replicateM n (diff l x)
  return ((sum xs) / (fromIntegral n))

stdN n l x = do
  xs <- replicateM n (diff l x)
  return (std xs)

mean xs = (sum xs) / (fromIntegral (length xs))
std xs = sqrt ((sum (map (\x -> (x - mean xs) ^ 2) xs)) / (fromIntegral (length xs)))

-- Binomial:
binom_pruned :: (Floating r, ADEV p m r s) => Int -> r -> s m Int
binom_pruned n p =
  if n == 0 then
    return 0
  else do
    n <- binom_pruned (n-1) p
    a <- flip_pruned p
    return (if a then n + 1 else n)

-- Geometric:
geom_pruned :: (Floating r, ADEV p m r s) => r -> s m Int
geom_pruned p = do
  b <- flip_pruned p
  if b then do
    n <- geom_pruned p
    return (n + 1)
  else
    return 1

geom_adev :: (Floating r, ADEV p m r s) => r -> p m Int
geom_adev p = do
  b <- flip_reinforce p
  if b then do
    n <- geom_adev p
    return (n + 1)
  else
    return 1

geom_adev_bernoulli :: (Floating r, ADEV p m r s) => r -> p m Int
geom_adev_bernoulli p = do
  b <- run_pruned (flip_pruned p)
  if b then do
    n <- geom_adev_bernoulli p
    return (n + 1)
  else
    return 1

geom_reversed_pruned :: (Floating r, ADEV p m r s) => r -> m r
geom_reversed_pruned p = expect_pruned (fmap fromIntegral helper)
  where 
  helper = do
    b <- flip_pruned p
    if not b then do
      n <- helper
      return (n + 1)
    else
      return 1

geom_reversed_adev :: (Floating r, ADEV p m r s) => r -> m r
geom_reversed_adev p = expect (fmap fromIntegral helper)
  where
  helper = do
    b <- flip_reinforce p
    if not b then do
      n <- helper
      return (n + 1)
    else
      return 1

geom_reversed_pruned_adev :: (Floating r, ADEV p m r s) => r -> m r
geom_reversed_pruned_adev p = expect (fmap fromIntegral helper)
  where
  helper = do
    b <- run_pruned (flip_pruned p)
    if not b then do
      n <- helper
      return (n + 1)
    else
      return 1

geom_weak :: (Floating r, ADEV p m r s) => r -> m r
geom_weak p = expect (fmap fromIntegral helper)
  where
  helper = do
    b <- flip_weak p
    if not b then do
      n <- helper
      return (n + 1)
    else
      return 1

other_demo = do
  x <- sampler $ diff (geom_weak) 0.4
  print x

geom_demo = do
  geom_triple <- sampler $ stochastic_derivative (geom_pruned $ bundle 0.4 1)
  putStr "Sampled (X, Y, w) for Geom(0.4): "
  print geom_triple
  binom_triple <- sampler $ stochastic_derivative (binom_pruned 10 $ bundle 0.4 1)
  putStr "Sampled (X, Y, w) for Binom(10, 0.4): "
  print binom_triple
  putStr "Estimated derivative for Geom(0.4) [1000 samples]: "
  geom_diff <- sampler $ diffN 1000 (expect_pruned . fmap fromIntegral . geom_pruned) 0.4
  print geom_diff
  binom_diff <- sampler $ diffN 1000 (expect_pruned . fmap fromIntegral . binom_pruned 10) 0.4
  putStr "Estimated derivative for Binom(10, 0.4) [1000 samples]: "
  print binom_diff

-- Random walk demo from Gaurav et al. 2022 (https://github.com/gaurav-arya/StochasticAD.jl/blob/main/tutorials/random_walk/core.jl)
walk :: (Floating r, ADEV p m r s) => Int -> r -> s m r
walk n p = do
  x <- walkFrom 0 n
  return (x * x)
  where
    walkFrom x 0 = return x
    walkFrom x n = do
      move_right <- flip_pruned (exp (-x / p))
      walkFrom (if move_right then x + 1 else x - 1) (n - 1)

walk_loss :: (Floating r, ADEV p m r s) => Int -> r -> m r
walk_loss n p = expect_pruned (walk n p)

walk_demo = do
  putStr "Estimated derivative for walk_loss(100) [1000 samples]: "
  walk_diff <- sampler $ diffN 1000 (walk_loss 100) 100
  print walk_diff

-- Toy program from Gaurav et al. 2022 (https://github.com/gaurav-arya/StochasticAD.jl/blob/main/tutorials/toy_optimizations/intro.jl)
toy_loss p = expect_pruned $ do
  let a = p * (1 - p)
  b <- binom_pruned 10 p
  x <- flip_pruned p
  let c = 2.0 * (fromIntegral b) + 3.0 * (if x then 1.0 else 0.0)
  r <- normal_pruned (fromIntegral b) a
  return (-a * c * r)

toy_demo = do
  putStr "Optimization trajectory of toy program (1000 steps, 20 evals per step):"
  vs <- sampler $ sgd toy_loss 0.001 0.5 1000
  print vs

sgd :: MonadDistribution m => (ForwardDouble -> m ForwardDouble) -> Double -> Double -> Int -> m [Double]
sgd loss eta x0 steps = 
  if steps == 0 then
    return [x0]
  else do
    -- use 20 evals per step
    v <- diffN 20 loss x0
    let x1 = x0 - eta * v
    xs <- sgd loss eta x1 (steps - 1)
    return (x0:xs)

main :: IO ()
main = do
  other_demo
  toy_demo
  geom_demo
  walk_demo