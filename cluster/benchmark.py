from timeit import default_timer as timer

import tensorflow as tf
import os
import sys

ava_root = sys.argv[3]

ava_info_path = os.path.join(ava_root, 'ava_info.pklz')
ava_images = os.path.join(ava_root, 'images/images')

image_list = os.listdir(ava_images)
image_paths = {os.path.join(ava_images, image) for image in image_list if os.path.isfile(os.path.join(ava_images, image))}
image_paths = list(image_paths - {'729377', '179118', '230701', '277832', '371434', '440774'})

##########################################################################################################################
###  UTILITY FUNCTIONS
##########################################################################################################################
def time_function(func, results_filename, task_name):
    def time_function_with_results_file(*args, **kwargs):
        start = timer()
        func(*args, **kwargs)
        end = timer()
        time = end - start

        with open(results_filename, 'a') as f:
            print(f"Time for {task_name}: {time:.2f}", file=f)

        return time
    return time_function_with_results_file

def tf_just_read_images(filename):
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=3)
    return image

def tf_resize_and_add_one(filename):
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (224, 224, 3))
    image += 1
    return image

##########################################################################################################################
###  TASK FUNCTIONS
##########################################################################################################################
def tasks_123(io_option, iterations):
    if io_option == 'python':
        for _ in range(iterations):
            for filename in image_paths:
                with open(filename, 'r') as f:
                    print(f)

    elif io_option == 'tf':
        for _ in range(iterations):
            dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(tf_just_read_images)
            for elem in dataset:
                print(elem)

def task4():
    dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(tf_resize_and_add_one)
    for elem in dataset:
        print(elem)

def task5(optimize_io):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    if (optimize_io):
        dataset = dataset.map(tf_resize_and_add_one, num_parallel_calls=tf.data.AUTOTUNE).batch(64).prefetch(-1).cache()
    else:
        dataset = map(tf_resize_and_add_one).batch(64)
    
    # Print the dataset twice to test the effect of optimizations
    for _ in range(2):
        for elem in dataset:
            print(elem)

def tasks_67(num_images):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths[:num_images]).map(tf_resize_and_add_one, num_parallel_calls=tf.data.AUTOTUNE).batch(64).prefetch(-1).cache()
    
    # Print the dataset twice to test the effect of cache depending on whether the dataset fits or not in RAM
    for _ in range(2):
        for elem in dataset:
            print(elem)
    
##########################################################################################################################
###  MAIN
##########################################################################################################################

'''
Ejecuta el benchmark de acceso a memoria.
Argumentos (pasados por linea de comandos):
    - Primer argumento: el fichero en el que guardar los resultados, ya sea para almacenar los del NFS o los del SSD.
    - Segundo argumento: si se va a hacer o no acceso en paralelo a los datos. En este caso, sólo se ejecuta una de las pruebas
    - Tercer argumento: la ruta donde está AVA
'''
def main():
    results_file = sys.argv[1]
    parallel_run = sys.argv[2]

    # TAREA 3: Acceso en paralelo a los datos. Sólo hay que hacer algo sencillo en este caso.
    # Si no se quiere hacer esta prueba, hacer el resto
    if (parallel_run == 'true'):
        time_function(tasks_123, results_file, 'Task_3')('python', 1)

    else:
        # TAREA 1: Cargar el dataset 1 vez (Python y TF)
        time_function(tasks_123, results_file, 'Task_1_python')('python', 1)
        time_function(tasks_123, results_file, 'Task_1_tf')('tf', 1)

        # TAREA 2: Cargar el dataset varias veces (Python y TF)
        time_function(tasks_123, results_file, f'Task_2_python_10_iterations')('python', 10)
        time_function(tasks_123, results_file, f'Task_2_tf_10_iterations')('tf', 10)

        # TAREA 4: Tarea simple con el dataset (TF)
        time_function(task4, results_file, 'Task_4')()

        # TAREA 5: Acceso en minibatches (más y menos optimización de I/O)
        time_function(task5, results_file, 'Task_5_unoptimized')(False)
        time_function(task5, results_file, 'Task_5_optimized')(True)

        # TAREA 6: Cargar todo el dataset en RAM
        time_function(tasks_67, results_file, 'Task_6')(50000)

        # TAREA 7: Cargar parte del dataset en RAM
        time_function(tasks_67, results_file, 'Task_6')(255000)


if (__name__ == '__main__'):
    main()