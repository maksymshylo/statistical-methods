# Statistical Methods of Pattern Recognition
Laboratory work from the university course.

## Setup

To run these applications you need to have **Python3.10**.

1. Clone repo:
```bash
git clone https://github.com/maksymshylo/computational_geometry.git
```
2. Create virtual environment.
```bash
python3.10 -m venv .venv
```
3. Activate it
```bash
source .venv/bin/activate
```
4. Install requirements:
```bash
pip install -r requirements.txt
```

## Lab 1 - Recognition of a noised image.

### Description
> The program converts a string to noised image and then decodes it.

### Usage
```commandline
 $ python3 lab1/decode_string.py --help
usage: decode_string.py [-h] --input_string INPUT_STRING --noise_level NOISE_LEVEL [--seed SEED]

options:
  -h, --help            show this help message and exit
  --input_string INPUT_STRING
                        input string
  --noise_level NOISE_LEVEL
                        noise level of bernoulli distribution
  --seed SEED           seed to debug
```

### Examples
```bash
python3 lab1/generate_string.py --input_string "billy herrington" --noise_level 0.35 --seed 45
```
Decoded string:  "billy herrington"

| Original image                        |           Noised image           | Decoded image                     |
|---------------------------------------|:--------------------------------:|-----------------------------------|
| ![](.imgs/lab1/test1/input_image.png) | ![](.imgs/lab1/test1/noised_image.png) | ![](.imgs/lab1/test1/decoded_image.png) |

```bash
python3 lab1/generate_string.py --input_string "billy herrington" --noise_level 0.45 --seed 45
```
Decoded string:  "nde deauff sc"

| Original image                        |              Noised image              | Decoded image                           |
|---------------------------------------|:--------------------------------------:|-----------------------------------------|
| ![](.imgs/lab1/test2/input_image.png) | ![](.imgs/lab1/test2/noised_image.png) | ![](.imgs/lab1/test2/decoded_image.png) |


## Lab 2 - Recognition of black vertical and horizontal lines using Gibbs Sampling
#### Examples

```bash
python3 lab2/gibbs_sampler_grid.py height_of_image width_of_image number_of_generated_lines noise_level column_probability number_of_iterations

python3 lab2/gibbs_sampler_grid.py 100 100 20 0.33 0.5 100
python3 lab2/gibbs_sampler_grid.py 50  50  10 0.33 0.5 100
```
##### `Precise Solution`

```bash
python3 lab2/gibbs_sampler_grid.py height_of_image width_of_image number_of_generated_lines noise_level column_probability

python3 lab2/precise_solution.py 20 20 5 0.3 0.5
python3 lab2/precise_solution.py 10 10 3 0.2 0.5
```

## Lab 3 - Gibbs Sampler for recognizing a noised string over another one

> Note: Lengths of string should be the same.

#### Examples
```bash
python3 row_over_row.py input_string_1 input_string_2 noise_level number_of_iterations


python3 row_over_row.py 'row'   'owr'   0.2  10
python3 row_over_row.py 'swap'  'paws'  0.33 30
python3 row_over_row.py 'hello' 'world' 0.2  20
```
##### some other examples of possible input strings with the same widths:
```
deliver <=> reviled
animal <=> lamina
depots <=> stoped
diaper <=> repaid
drawer <=> reward
looter <=> retool
murder <=> redrum
redips <=> spider
debut <=> tubed
deeps <=> speed
peels <=> sleep
serif <=> fires
steel <=> leets
````
