# Make all the images 128x128
python crop_images.py

# Train a classifier for attribute 9 (Blond_Hair), protected attribute is
# included to keep track of certain attributes.
python main.py --experiment baseline --attribute 9 --protected_attribute 20

# Train for attribute 20 (gender) with protected attribute 9
python main.py --experiment baseline --attribute 20 --protected_attribute 9

# Generate new images from the Latent Vectors z
python generate_images.py --experiment orig --attribute 9 --protected_attribute 20

# Classify each of the the newly generated images for blond or not blonde.
python get_scores.py --attribute 9

# Classify each of the newly generated images for gender
python get_scores.py --attribute 20

# Estimate Hyperplanes and compute complimentary latent vectors z'
python linear.py

# Generate Images from z'
generate_images.py --experiment pair

# Train models with augmented datasets
main.py --experiment model
