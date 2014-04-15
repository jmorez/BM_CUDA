Written by XXXXXXX (University of XXXXXX) as a part of a bachelor's 
thesis that uses a blockmatching algorithm  to gather a statistical 
population for denoising single pixels in an image.

This function contains the findMatches CUDA kernel that will find 
matches for every pixel in an image of size M by N (rows x columns).
The algorithm is based on the following paper:
http://www.mia.uni-saarland.de/Publications/zimmer-lnla08.pdf


Throughout this code, I will insert footnotes inside comments of the 
format (#) which -unsurprisingly- can be found at the bottom. This is 
first of all to keep the code compact, but also to allow both the 
reader (you) as the developer (me) to understand this code better, i.e. "why 
use datatype X", "why do loop Y like this", "why is thisindex Z minus 
one", ...  

