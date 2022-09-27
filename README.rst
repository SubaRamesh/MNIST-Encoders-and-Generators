Autoencoder
------------
Train autoencoder to encode MNIST digits to 2 dimensional latent space. This space can then be visualized as below by label to show clustering of digits and find similar features in digits. 

Example:
.. image:: figures/3_3_plot.png
    :alt: Plot 2 dimensional encoding
    :height: 300

For this encoding, the reconstruction currently looks like below:

.. image:: figures/3_3_original.png
    :alt: Original digit image
    :height: 300

.. image:: figures/3_3_reconstructed.png
    :alt: Reconstructed digit image
    :height: 300
    

Future idea: sampling latent space to generate a greater variety of digits -> viable method to generate data?


GAN
------------
Use adversial algorithm to generate images of clothing by learning from the Fashion MNIST dataset. 

Example:
.. image:: figures/CGAN_Generations_Epoch_23_3.png
    :alt: Set of generated images
    :height: 300

Originally, GAN was trained without labels. This shows that the generator learned how to create jackets very well but learned very little about other clothing. To remedy this, I added a label feature to the generator so we can give the generator a specific clothing to create. We will query the label space randomly during training. 

Example:
.. image:: figures/CGAN_Generations_Epoch_23_labels.png
    :alt: Set of generated images with labels
    :height: 300
    
While the images are now lower quality, there is a higher diversity in the images. This time the generator learned how to create different types of clothing and also label them properly. Looking at the images, the labels seem to be accurate.

