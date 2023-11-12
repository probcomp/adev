{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use bimap" #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
module Main (main) where

import Numeric.ADEV.Class
import Data.Map (Map, singleton, (!), delete, member)
import Numeric.Log (Log(..))
import Numeric.ADEV.Reverse ()
import Numeric.AD.Mode.Reverse ( jacobian )
import Control.Monad.Bayes.Sampler.Lazy (sampler)

-- Import Reverse from ad
import Numeric.AD.Internal.Reverse (Reverse, Tape)
import Data.Reflection (Reifies)

-- Values can be in traces
data Value r =
    R r    |
    B Bool |
    I Int

class Traceable a r where
    asValue :: a -> Value r
    fromValue :: Value r -> a

instance Traceable Double Double where
    asValue = R
    fromValue (R r) = r
    fromValue _ = error "Type error"

instance (Reifies s Tape) => Traceable (Reverse s Double) (Reverse s Double) where
    asValue = R
    fromValue (R r) = r
    fromValue _ = error "Type error"

instance Traceable Bool r where
    asValue = B
    fromValue (B b) = b
    fromValue _ = error "Type error"

instance Traceable Int r where
    asValue = I
    fromValue (I i) = i
    fromValue _ = error "Type error"

-- A trace maps string-valued names to values
-- using a Map.
type Trace r = Map String (Value r)

newtype G m r a = G (m (Trace r, Log r, a), Trace r -> m (Log r, a, Trace r))

newtype Dist m r a = Dist (m (a, Log r), a -> m (Log r))

simulateD :: Dist m r a -> m (a, Log r)
simulateD (Dist d) = fst d

densityD :: Dist m r a -> a -> m (Log r)
densityD (Dist d) = snd d

simulate_ :: G m r a -> m (Trace r, Log r, a)
simulate_ (G g) = fst g

density_  :: G m r a -> Trace r -> m (Log r, a, Trace r)
density_ (G g) = snd g

simulate :: Monad m => G m r a -> m (Trace r, Log r)
simulate g = do
    (t, c, _) <- simulate_ g
    return (t, c)

density :: (Monad m, RealFloat r) => G m r a -> Trace r -> m (Log r)
density g t = do
    (d, _, t') <- density_ g t
    return (if null t' then d else 0)

instance Functor m => Functor (G m r) where
    fmap f (G g) = G (fmap (\(a,b,c) -> (a, b, f c)) (fst g), fmap (\(a,b,c) -> (a, f b, c)) . snd g)

instance (Monad m, RealFloat r) => Applicative (G m r) where
    pure a = G (pure (mempty, 1, a), \t -> return (1, a, t))
    f <*> g = f >>= (<$> g)

instance (Monad m, RealFloat r) => Monad (G m r) where
    return = pure
    m >>= f = G (bind_sim, bind_dens)
        where
            bind_sim = do
                (t, c, a)   <- simulate_ m
                (t', c', b) <- simulate_ (f a)
                return (t <> t', c * c', b)
            bind_dens t = do
                (c, a, t') <- density_ m t
                (c', b, t'') <- density_ (f a) t'
                return (c * c', b, t'')

sample :: (Monad m, RealFloat r, Traceable a r) => Dist m r a -> String -> G m r a
sample d name = G (sample_simulate, sample_density)
    where
        sample_simulate = do
            (a, c) <- simulateD d
            return (Data.Map.singleton name (asValue a), c, a)
        sample_density t = do
            if name `Data.Map.member` t then do
                let x = fromValue $ t Data.Map.! name
                c <- densityD d x
                return (c, x, Data.Map.delete name t)
            else
                return (0, error "Type error", t)

factor :: (Monad m) => Log r -> G m r ()
factor c = G (return (mempty, c, ()), \t -> return (c, (), t))


observe :: Monad m => Dist m r a -> a -> G m r ()
observe d x = G (observe_simulate, observe_density)
    where
        observe_simulate = error "Not implemented"
        observe_density t = do
            c <- densityD d x
            return (c, (), t)

makePrimitive sampler pdf = Dist (s, d)
    where
        s = do {x <- sampler; return (x, pdf x)}
        d = return . pdf

flip_pdf :: (RealFloat r) => r -> Bool -> Log r
flip_pdf p b = Exp $ log $ if b then p else 1 - p

flipENUM :: (ADEV p m r, RealFloat r) => r -> Dist (p m) r Bool
flipENUM p = makePrimitive (flip_enum p) (flip_pdf p)

flipREINFORCE :: (ADEV p m r, RealFloat r) => r -> Dist (p m) r Bool
flipREINFORCE p = makePrimitive (flip_reinforce p) (flip_pdf p)

flipWEAK :: (ADEV p m r, RealFloat r) => r -> Dist (p m) r Bool
flipWEAK p = makePrimitive (flip_weak p) (flip_pdf p)

normal_pdf :: (RealFloat r) => r -> r -> r -> Log r
normal_pdf mu sigma x = Exp $ log (1 / (sigma * sqrt (2 * pi))) - 0.5 * ((x - mu) / sigma)^2

normalREPARAM :: (ADEV p m r, RealFloat r) => r -> r -> Dist (p m) r r
normalREPARAM mu sigma = makePrimitive (normal_reparam mu sigma) (normal_pdf mu sigma)

normalREINFORCE :: (ADEV p m r, RealFloat r) => r -> r -> Dist (p m) r r
normalREINFORCE mu sigma = makePrimitive (normal_reinforce mu sigma) (normal_pdf mu sigma)

circleExampleModel :: (RealFloat a, Traceable a a, ADEV p m a) => a -> G (p m) a ()
circleExampleModel r = do
    x <- sample (normalREPARAM 0 1) "x"
    y <- sample (normalREPARAM 0 1) "y"
    observe (normalREPARAM (sqrt (x^2 + y^2)) 0.1) r

circleExampleGuide :: (RealFloat b, Traceable b b, ADEV p m b) => [b] -> G (p m) b (b, b)
circleExampleGuide [mu1, mu2, logsigma1, logsigma2] = do
    x <- sample (normalREPARAM mu1 (exp logsigma1)) "x"
    y <- sample (normalREPARAM mu2 (exp logsigma2)) "y"
    return (x, y)

-- Actually a negative ELBO
elbo :: (ADEV p m r, RealFloat r, Traceable r r) => [r] -> m r
elbo params = expect $ do
    (x, q) <- simulate (circleExampleGuide params)
    p      <- density  (circleExampleModel 4) x
    return (-(ln p - ln q))

initialParams :: Num r => [r]
initialParams = [0, 0, 0, 0]

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
    vs <- sampler $ sgd elbo 0.001 initialParams 500
    print (fmap (\[x,y,z,w] -> ((x,y), (exp z, exp w))) vs)