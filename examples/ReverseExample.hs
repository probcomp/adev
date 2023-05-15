{-# LANGUAGE  FlexibleContexts, RankNTypes, AllowAmbiguousTypes #-}

module Main (main) where

import Numeric.ADEV.Class
import Numeric.ADEV.Reverse ()
--import Numeric.ADEV.Interp ()
import Numeric.AD.Mode.Reverse
import Control.Monad.Bayes.Class (MonadDistribution)
import Control.Monad.Bayes.Sampler.Lazy (sampler, Sampler)
import Control.Monad.Cont (ContT(..))
import Numeric.AD.Internal.Reverse
--import Numeric.AD.Mode
--import Numeric.AD.Jacobian
import Data.Reflection

-- l :: ADEV p m r s => r -> m r
l :: (Reifies s Tape) => Reverse s Double -> Sampler (Reverse s Double)
l theta = expect $ do
    b <- flip_reinforce theta
    if b then
        return 0
    else
        return (-theta * 0.5)

--sgd :: (Reifies s Tape, ADEV p m (Reverse s Double) g) => (Reverse s Double -> m (Reverse s Double)) -> Double -> Double -> Int -> m [Double]

sgd :: (forall s. (Reifies s Tape) => Reverse s Double -> Sampler (Reverse s Double)) -> Double -> Double -> Int -> Sampler [Double]
sgd loss eta x0 steps = 
    if steps == 0 then
        return [x0]
    else do
        v <- diffF loss x0
        let x1 = x0 - eta * v
        xs <- sgd loss eta x1 (steps - 1)
        return (x0:xs)

main :: IO ()
main = do 
    vs <- sampler $ sgd l 0.2 0.2 100
    print vs