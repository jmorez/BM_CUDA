Written by Jan Morez (University of Antwerp) as a part of <a href="https://dl.dropboxusercontent.com/u/17216535/Poster.pdf" target="_blank">a bachelor's 
thesis </a> that uses a blockmatching algorithm  to gather a statistical 
population for denoising single pixels in an image.

This function contains the findMatches CUDA kernel that will find 
matches for every pixel in an search window of size M by N.
The algorithm is based on the following paper:
http://www.mia.uni-saarland.de/Publications/zimmer-lnla08.pdf


