# numc

Here's what I did in project 4:
- After reading the project 4 specs we went on zoom and started working on naive functions for the matrices. 
After finishing the naive functions in matrix.c. we moved on to completing finishing the numc.c file.
- In our numc.c file, we first worked on the number method *Matrix61c_add.  We made sure to handle these error cases: allocation failure (which is runtime error), invalid arguments (type error), and different dimensions (value error). Then, we changed the return value’s mat to our result, and made sure to update the shape. We handled the subtraction method similarly. 
- For *Matrix61c_multiply method, we made sure to handle these errors: invalid arguments (type error), allocation fails (runtime error), and different dimensions (value error). For *Matrix61c_pow, we had these error checks: allocation failed (runtime error), the power should be an integer and non negative (type error, value error), and that the matrix should be a square since only square matrices can be powered (value error). 
- In order to test the above functions, we realized that we needed to implement indexing. In our *Matrix61c_subscript method, we had to check whether our key is a slice, int or a tuple, because all three of them will be handled differently. For a key that is a tuple. We had to use the PyArg_UnpackTuple function to check if the tuple are both longs. If they are both longs and are at valid indices, then we can return our resulting value, if they are out of range, we will be raising an index error. If one key is a slice and the other is a long, we first need to check if the long is larger than the number of rows, or a negative value since that will be out of range. Then, we have a start, end, step, and step length for our error checking with the PySlice_GetIndicesEx as a flag. We check for invalid arguments (type error), invalid slice information when the step length is 0 or step is not one (value error), and of course invalid indexes (index error). 
- For the case where the key is a slice, we have a case where the matrix is 1D, and when it is not. We first check its arguments with the PySlice_GetIndicesEx as a flag, and then also check the step length and steps again (value error), and then call allocate matrix ref function which will return a negative one if there is an error. For 2D matrices, we perform the same checks (the invalid arguments, step length and step, and invalid indices).
- For the int case, we check again if we are dealing with a 1D matrix. If we are, we have another check if the row is one or if the column is one, and then we are able to call allocate_matrix_ref with the correct dimensions. In both cases (1D, 2D), we still check whether the index is valid or not. The only difference is the values passed into our allocate_matrix_ref function.
Next, we implemented the Matrix61c_set_subscript function. This was similar to the earlier subscript function, but there were a few extra things to implement. We had to check whether the value we want to insert is a list, long, float, or some other type. If what we want to insert is an int or a float, and where we are inserting is of the same type, then we can simply double index into the data double pointer and set it as our value. If none of these apply, we have to check if the value is a list,, and then perform some more error checks in case the list we received surpassed our column size or row size. Then, we can implement a for loop to replace each i in our current data[0][i] or data[i][0] to each item in the given list. 
- We then worked on our instance methods, with the set_value and get_value methods. For our set method, we had to unpack the tuple and perform checks that made sure that all our arguments were valid (type error), and that our indexes were valid as well (index error). If they are valid longs and within our rows and columns, then we call our set function from matrix.c. We made sure to use Py_RETURN_NONE when we successfully set the index to that value. - For our get instance method, we also had to parse our tuple into row and col which are ints (longs). Then, we checked again that they are both valid indexes (index error). If they are all valid and both valid longs, then we use PyFloat_FromDouble to return the value at that index. 
- Having completed the indexing methods, we tested our number methods with the commands given on the website. Then, we realized that we also needed to write our own tests, so we added the medium, large, and of course all the error checks that the functions are supposed to handle properly. For the error checking portion, we were unsure of how to ensure that the function calls the error, so we looked on piazza to see that a teaching assistant/instructor has posted an example of a test error. Therefore, we used that as reference for most of our error tests. 
- We then moved on to our setup.py file, which integrates our c files with python for our numc. We referenced the linked documentation and followed the examples on it. There was some confusion so we asked one of the instructors and peers for assistance. Eventually it managed to work, and we are now able to use numc on our terminal (yay!).
Now onto the speeding up portion, we first implemented our matrix in a 2D way, but we later changed the format of our allocate matrix, allocate matrix ref, and deallocate. We decided to change it to row-major order, but also keep the same 2D structure. We did this by first assigning row times column space for our 1D structure, and then having another array of pointers that point to each new row. 
- For the deallocate function, we made sure to handle these two cases: when the matrix’s ref count is one, and otherwise. We had a double freeing issue, but fixed it by only freeing data[0] of our matrix’s data if it is at ref count one. We figured out that looping through data[i] and freeing them caused the double free error. 
- For the get and set functions, since we only changed the back end of our matrix to 1D, we are still able to access it like a 2D matrix.
- We had tried OpenMP to speed up our fill_matrix, add_matrix, and sub_matrix functions. We thought this might help because we currently have a for loop that indexes into our data and we are able to use this directive #pragma omp for  that divides loop iterations between the spawned threads.
- For mul_matrix, we had quite a bit of trouble attempting to speed it up. We tried unrolling, simd, openMP. Unfortunately, unrolling alone did not really help much with the performance. We saw that transposing the second matrix and simd vectorizing both matrices will help, so that was what we did. When transposing, we used openmp to speed it up a little. We have also implemented cache blocking in our two innermost for-loops with block size of 2 and 8, and it worked on mul, speeding it up by almost twice. 
- For pow_matrix, we had decided to implement openmp, reduced squaring, and using mul_matrix. We also thought of adding a naive case for matrices under size 500 since we had read that this would improve performance, but the numbers in the autograder seem to say otherwise and our pow speedup decreased, while our overall performance increased. We still think that this might be because of the smaller cases, so now we are trying to handle small cases with the naive solution instead of our repeated squaring.
