module Control.Monad.Bayes.Sampler.Lazy.Coupled where

import Control.Monad.Bayes.Sampler.Lazy

-- TODO: smarter coupling?
-- Would require a sampler that does smarter divying up of 
-- randomness. Ideally we would not be so sensitive to 
-- changes like "a => do { x <- a; return x }", which are
-- supposed to be meaning-preserving.
-- User-named choices are of course one way to do this;
-- Wingate has something a little more automated. 
coupled :: Sampler a -> Sampler b -> Sampler (a, b)
coupled (Sampler s1) (Sampler s2) = Sampler $ \g -> (s1 g, s2 g)