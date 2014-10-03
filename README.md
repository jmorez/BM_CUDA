<h4>Context</h4>
<p>
Written by Jan Morez (University of Antwerp) as a part of <a href="https://dl.dropboxusercontent.com/u/17216535/Poster.pdf" target="_blank">a bachelor's 
thesis </a> that uses a blockmatching algorithm  to gather a statistical 
population for denoising single pixels in an image.</p>

<h4>What is in this repository?</h4>
<p>The findMatches CUDA kernel (along with Matlab helper functions) that will find 
matches for every pixel in an search window of size M by N.
The algorithm is based on the following paper:
http://www.mia.uni-saarland.de/Publications/zimmer-lnla08.pdf</p>

<h4>Usage:</h4>
<p>The src/C/findMatches.cu file will have to be compiled into a .mex file. 
The comments in the .cu file will tell you which other inputs are required. 
If you intend on using this code and are confused because of the horrible documentation,
you can contact me through jan.morez AT gmail DOT com</p>

