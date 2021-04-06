import sys, os, argparse
import json, string
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from decimal import *

EPS = 10**-320

def get_bigrams(json_path, alphabet_list):

    if not os.path.isfile(json_path):
        raise Exception('frequencies.json does not exist')

    with open(json_path) as json_file: 
        frequencies_dict = json.load(json_file)
        
    alphabet_dict = dict((j,i) for i,j in enumerate(alphabet_list))
    # all pairs as array
    array = np.zeros([len(alphabet_list),len(alphabet_list)]).astype('int')
    for i in alphabet_dict:
        for j in alphabet_dict:
            if i + j in frequencies_dict:
                array[alphabet_dict[i]][alphabet_dict[j]] = frequencies_dict[i+j]

    # make a-priopi probabilities from frequencies
    p_k = (array.T/array.sum(axis=1)).T

    # make np.log(EPS) where p(k) = 0
    p_k = np.log(p_k, out=np.full_like(p_k, np.log(EPS)), where=(p_k!=0))

    return p_k


def import_images(folder_path, alphabet_list):
    """
    reads images from folder_path and creates set of images
    """
    reference_images = {}
    for i in alphabet_list[:-1] + ['space']:
        img = np.array(Image.open(folder_path + f'/{i}.png')).astype('int')
        reference_images[i] = img
    reference_images[' '] = reference_images.pop('space')
    return reference_images


def string_to_image(string, reference_images):
    """
    convertes string to image
    """
    # create string as array
    image = reference_images[string[0]]
    for i in string[1:]:
        image = np.hstack([image,reference_images[i]])
    return image


def apply_noise(input_image, noise_level):
    """
    apply bernoulli noise to image
    """
    n,m = input_image.shape
    # generate binomial noise
    ksi = np.random.binomial(size=n*m, n=1, p=noise_level).reshape(input_image.shape)
    noised_image = ksi^input_image
    return noised_image


def calculate_tail(input_image, p, letters, p_k, mask):
    """
    calculates sum of probabilities for the rest of the string
    """
    n = input_image.shape[1]
    # array of sums of probabilities
    sum_of_probab = np.full((n+1),0,dtype=Decimal)
    # last element always equals 1
    sum_of_probab[-1] = 1
    for s in range(n-1,-1,-1):
        inner_sum = np.full((len(letters)),Decimal(0).normalize(),dtype=Decimal)
        for i,letter in enumerate(letters):
            # the end of possible letter
            cut_till = s + letter.shape[1]
            # calculate conditional probability
            if cut_till <= n:
                x = input_image[:,s:cut_till]
                # multiplication on mask is needed to sum in cond_probab only
                # that pixels which are black in image at previous iteration 
                if p == 0:
                    xor_value = ((x^letter)*np.log(p + EPS))*mask[:,s:cut_till]
                elif p == 1:
                    xor_value = ((1^x^letter)*np.log(EPS))*mask[:,s:cut_till]
                else:
                    xor_value = ((x^letter)*np.log(p) + (1^x^letter)*np.log(1-p))*mask[:,s:cut_till]
                
                cond_probab = np.sum(xor_value)  + p_k[i]
                # add to inner sum
                inner_sum[i] = Decimal(cond_probab).exp()*Decimal(sum_of_probab[cut_till])
        sum_of_probab[s] = np.sum(inner_sum)
    return sum_of_probab


def letter_probab(input_image, p, letters, sum_of_probab, p_k, mask):
    """
    calculates probability of the letter p(x)
    """
    n = input_image.shape[1]
    letter_probab = np.zeros(len(letters),dtype=Decimal)
    for i,letter in enumerate(letters):
        cut_till = letter.shape[1]
        if cut_till <= n:
            x = input_image[:,:cut_till]
            # multiplication on mask is needed to sum in cond_probab only
            # that pixels which are black in image at previous iteration 
            if p == 0:
                xor_value = ((x^letter)*np.log(p + EPS))*mask[:,:cut_till]
            elif p == 1:
                xor_value = ((1^x^letter)*np.log(EPS))*mask[:,:cut_till]
            else:
                xor_value = ((x^letter)*np.log(p) + (1^x^letter)*np.log(1-p))*mask[:,:cut_till]

            cond_probab = np.sum(xor_value)  + p_k[i]
                
            letter_probab[i] = Decimal(cond_probab).exp()*Decimal(sum_of_probab[cut_till])
    return letter_probab


def generete_string(input_image, p, letters, bigrams, alphabet_list, mask):
    """
    generate string from image
    """
    output_string = []
    width_prev_letter = 0
    
    p_k = bigrams[-1,:]
    while input_image[:,width_prev_letter:].size != 0:
        # cut input_image based on previous letter
        input_image = input_image[:,width_prev_letter:]
        # cut mask based on input_image
        mask = mask[:,width_prev_letter:]
        # calculate probabilities for the first letter
        sum_of_probab = calculate_tail(input_image,p,letters,p_k,mask)
        pk1 = letter_probab(input_image,p,letters,sum_of_probab,p_k,mask)
        # generate letter from probability
        generated_letter = np.random.choice(len(pk1), 1, p=list(pk1/pk1.sum()))[0]
        p_k = bigrams[generated_letter,:]
        width_prev_letter = letters[generated_letter].shape[1]
        # get generated letter
        output_string.append(alphabet_list[generated_letter])
    return ''.join(output_string)


def recognize_strings(noised_image, noise_level, letters, reference_images, bigrams, alphabet_list, n_iter):
    """
    recognize text where one string is over another one

    noised_image - image to recognize
    noise_level - level of bernoulli noise
    letters - array of binary images
    reference_images - dict of letters
    bigrams - pairs probabilities of letters
    alphabet_list - alphabet
    n_iter - number of iterations
    """

    # mask is needed to sum in generete_string only pixels which are not black
    # set mask for first iteration as array of ones 
    mask = np.ones_like(noised_image, dtype = int)
    for _ in range(n_iter):
        # generate output_string_1
        output_string_1 = generete_string(noised_image, noise_level, letters, bigrams, alphabet_list, mask)
        # convert it to image
        output_image_1 = string_to_image(output_string_1, reference_images)
        # update mask 
        mask = np.logical_not(output_image_1)
        # generate output_string_2
        output_string_2 = generete_string(noised_image, noise_level, letters, bigrams, alphabet_list, mask)
        # convert it to image
        output_image_2 = string_to_image(output_string_2, reference_images)
        # update mask 
        mask = np.logical_not(output_image_2)
    return output_string_1, output_image_1, output_string_2, output_image_2


def main():

    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("input_string_1", type=str,   help="first input string ")
    parser.add_argument("input_string_2", type=str,   help="second input string")
    parser.add_argument("noise_level",    type=float, help="noise level of bernoulli distribution")
    parser.add_argument("n_iter",         type=int,   help="number of iterations")

    args = parser.parse_args()


    alphabet_list = list(string.ascii_lowercase + ' ')
    reference_images = import_images("alphabet",alphabet_list)
    bigrams = get_bigrams('frequencies.json', alphabet_list)
    letters = list(reference_images.values())

    input_image_1 =  string_to_image(args.input_string_1,reference_images)
    input_image_2 =  string_to_image(args.input_string_2,reference_images)

    if input_image_1.shape[1] != input_image_2.shape[1]:
        raise Exception("input images must have the same size")

    input_image   =  np.logical_or(input_image_1,input_image_2)
    noised_image  =  apply_noise(input_image,args.noise_level)

    output_string_1, output_image_1, output_string_2, output_image_2 =  \
        recognize_strings(noised_image, args.noise_level, letters, \
                          reference_images, bigrams, alphabet_list, args.n_iter)

    print("first string:",output_string_1,"\nsecond string:", output_string_2)

    plt.imsave("input_image.png",    input_image,    cmap ="binary")
    plt.imsave("noised_image.png",   noised_image,   cmap ="binary")
    plt.imsave("output_image_1.png", output_image_1, cmap ="binary")
    plt.imsave("output_image_2.png", output_image_2, cmap ="binary")


if __name__ == "__main__":
    main()
