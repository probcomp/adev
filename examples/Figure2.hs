module Main (main) where

import Numeric.ADEV.Class
import Numeric.ADEV.Diff (diff)
import Numeric.ADEV.Interp ()
import Numeric.AD.Mode.Forward.Double (ForwardDouble)
import Control.Monad.Bayes.Class (MonadDistribution)
import Control.Monad.Bayes.Sampler.Lazy (sampler)

l :: ADEV p m r s => r -> m r
l theta = expect $ do
    b <- flip_reinforce theta
    if b then
        return 0
    else
        return (-theta / 2)

sgd :: MonadDistribution m => (ForwardDouble -> m ForwardDouble) -> Double -> Double -> Int -> m [Double]
sgd loss eta x0 steps = 
    if steps == 0 then
        return [x0]
    else do
        v <- diff loss x0
        let x1 = x0 - eta * v
        xs <- sgd loss eta x1 (steps - 1)
        return (x0:xs)

main :: IO ()
main = do 
    vs <- sampler $ sgd l 0.2 0.2 100
    print vs