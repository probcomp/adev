module Control.Monad.Bayes.Sampler.Lazy.Coupled where

import Control.Monad.Bayes.Sampler.Lazy

-- TODO: smarter coupling?
-- Would require a sampler that does smarter divying up of 
-- randomness. Ideally we would not be so sensitive to 
-- changes like "a => do { x <- a; return x }", which are
-- supposed to be meaning-preserving.
-- User-named choices are of course one way to do this;
-- Wingate has something a little more automated. 
coupled :: SamplerT a -> SamplerT b -> SamplerT (a, b)
coupled (SamplerT s1) (SamplerT s2) = SamplerT $ \g -> (s1 g, s2 g)

get_seed :: SamplerT Tree
get_seed = SamplerT $ \g -> g