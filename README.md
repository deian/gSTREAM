# gSTREAM: GPU (CUDA) implementation of eSTREAM Ciphers #

This is an implementation of CUDA eSTREAM-like framework. The code
is copied from my old Master's thesis svn repo (so certain things
might be missing -- if you find that this is the case, please let me
know) at the request of a few students working on a similar topic.
See the [thesis](http://www.deian.net/pubs/stefan:2011:analysis.pdf)
for high-level details of the implementations.

The implementation includes an CUDA implementations of Grain, HC128,
MICKEY, Rabbit, Salsa20, Trivium. Additionally an optimized (faster
than the eSTREAM version) C implementation of Grain is provided.
