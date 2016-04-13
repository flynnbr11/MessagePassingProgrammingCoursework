To compile the code, type 
	$ make
into the command line, which will generate the executable image. By typing
	$ ./image
you will run the main program (image_reconstruction.c). 

To test the code, various print statements have been commented out throughout to
demonstrate what the section of the code does. Uncomment and rerun to test these 
sections. 

To change the image processed, output file, convergence threshold or frequency 
of convergence checking, uncomment lines 59 to 66, and enter these values as
command line arguments. 


Separate test code is provided, which is simply the same code but with different
print statments. 
We can compare the results of two output images 
(provided they use the same input image and have the same values for convergence
and frequency), by typing 
	$ diff output_image_1.pgm output_image_2.pgm
into the command line, while in the same directory as the images produced. 



Included in the tar file are the following:

image_reconstruction.c	: The main code to process an image

pgmio.c; pgmio.h				: Required to read and write images
arraloc.c; arraloc.h		: Required for dynamic array allocation

given_images						: contains the images made available as samples

test_code.c							: The same program as image_reconstruction.c, but with
													more print statements included to demonstrate the 
													progress of the program
									
Makefile								: Can be used to compile image_reconstruction.c by 
													typing, into the command line:
					
