# Statistical Methods of Pattern Recognition
Labs for University Course     

## Lab 1 - Generate string of letters from noised input image
#### Examples
```bash
python3 generate_string.py input_string noise_level

cd lab1/

python3 generate_string.py 'some noised string' 0.2
python3 generate_string.py 'hello world' 0.35
```
## Lab 2 - Recognizing Black Vertical and Horizontal Lines
#### Examples
##### `Gibbs Sampler Solution`

```bash
python3 lab2/gibbs_sampler_grid.py height_of_image width_of_image number_of_generated_lines noise_level column_probability number_of_iterations

python3 lab2/gibbs_sampler_grid.py 100 100 20 0.33 0.5 100
python3 lab2/gibbs_sampler_grid.py 50 50 10 0.33 0.5 100
```
##### `Precise Solution`

```bash
python3 lab2/gibbs_sampler_grid.py height_of_image width_of_image number_of_generated_lines noise_level column_probability

python3 lab2/precise_solution.py 20 20 5 0.3 0.5
python3 lab2/precise_solution.py 10 10 3 0.2 0.5
```
## Lab 3 - Gibbs Sampler for recognizing string over another one with noise
#### Examples
```bash
python3 row_over_row.py input_string_1 input_string_2 noise_level number_of_iterations

cd lab3/

python3 row_over_row.py 'row' 'owr' 0.2 10
python3 row_over_row.py 'swap' 'paws' 0.33 30
python3 row_over_row.py 'stressed' 'desserts' 0.33 100
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
