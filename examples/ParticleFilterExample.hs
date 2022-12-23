module Main (main) where

import Numeric.ADEV.Class
import Numeric.ADEV.Diff (diff)
import Numeric.ADEV.Interp ()
import Numeric.ADEV.Distributions (normalD)
import Numeric.AD.Mode.Forward.Double (ForwardDouble)
import Control.Monad.Bayes.Class (MonadDistribution)
import Control.Monad.Bayes.Sampler.Lazy (sampler)
import Numeric.Log (Log(..))

--   smc :: ([a] -> Log r) -> D m r a -> (a -> D m r a) -> ([a] -> m r) -> Int -> Int -> m r

dens :: (RealFrac r, Floating r) => D m r Double -> Double -> Log r
dens (D _ f) x = f x 

normalDensity :: (RealFrac r, Floating r) => r -> r -> Double -> Log r
normalDensity mu sig x = Exp $ -log(sig) - log(2*pi) / 2 - ((realToFrac x)-mu)^2/(2*sig^2)

ys = [undefined, 1,2,3,4,5]

l :: (MonadDistribution m, RealFloat r, Floating r, ADEV p m r s) => r -> m r
l theta = smc p q0 q f 2 1000
  where
    p xs = let xys = zip (map realToFrac xs) (reverse (take (length xs) ys)) in
           pxys xys
    pxys [] = undefined
    pxys [(x, y)] = normalDensity 0 (exp theta) (realToFrac x)
    pxys ((x,y):((xprev,yprev):xys)) = normalDensity (realToFrac xprev) (exp theta) (realToFrac x) * normalDensity (realToFrac x) 1 y * pxys ((xprev,yprev):xys)
    q0 = normalD 0 (exp theta)
    q x = normalD (realToFrac x) (exp theta)
    f xs = return 1

sga :: MonadDistribution m => (ForwardDouble -> m ForwardDouble) -> Double -> Double -> Int -> m [Double]
sga loss eta x0 steps = 
    if steps == 0 then
        return [x0]
    else do
        v <- diff loss x0
        let x1 = x0 + eta * v
        xs <- sga loss eta x1 (steps - 1)
        return (x0:xs)

main :: IO ()
main = do
    vs <- sampler $ sga l 10.0 0.0 500
    print (vs)