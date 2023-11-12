module Numeric.ADEV.Distributions (normalD, geometricD) where

import Numeric.ADEV.Class ( D(..) )

import Control.Monad.Bayes.Class (
    MonadDistribution,
    geometric,
    normal)

import Numeric.Log (Log(..))

normalD :: (MonadDistribution m, RealFrac r, Floating r) => r -> r -> D m r Double
normalD mu sig = D (normal (realToFrac mu) (realToFrac sig)) (\x -> Exp $ -log sig - log (2*pi) / 2 - (realToFrac x-mu)^(2::Int)/(2*sig^(2::Int)))

geometricD :: (MonadDistribution m, RealFrac r, Floating r) => r -> D m r Int
geometricD p = D (geometric (realToFrac p)) (\x -> Exp $ log p + fromIntegral (x-1) * log (1-p))
