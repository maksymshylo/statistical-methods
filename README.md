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
