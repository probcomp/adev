{-# LANGUAGE  FlexibleContexts, RankNTypes, AllowAmbiguousTypes #-}

module Main (main) where

import Numeric.ADEV.Class ( ADEV(flip_reinforce, expect) )
import Numeric.ADEV.Reverse ()
--import Numeric.ADEV.Interp ()
import Numeric.AD.Mode.Reverse
import Control.Monad.Bayes.Sampler.Lazy (sampler)
import Numeric.AD.Internal.Reverse
--import Numeric.AD.Mode
--import Numeric.AD.Jacobian
import Data.Reflection

l :: ADEV p m r => [r] -> m r
l theta = expect $ do
    b <- flip_reinforce ((theta !! 0) * (theta !! 1))
    if b then
        return 0
    else
        return (-((theta !! 0) * (theta !! 1)) * 0.5)

sgd :: (Monad m) => (forall s. (Reifies s Tape) => [Reverse s Double] -> m (Reverse s Double)) -> Double -> [Double] -> Int -> m [[Double]]
sgd loss eta x0 steps =
    if steps == 0 then
        return [x0]
    else do
        v <- jacobian loss x0
        let x1 = zipWith (-) x0 (fmap (eta *) v)
        xs <- sgd loss eta x1 (steps - 1)
        return (x0:xs)

main :: IO ()
main = do
    vs <- sampler $ sgd l 0.2 [0.4, 0.5] 200
    print vs