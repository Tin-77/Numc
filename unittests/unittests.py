from utils import *
from unittest import TestCase

"""
For each operation, you should write tests to test  on matrices of different sizes.
Hint: use dp_mc_matrix to generate dumbpy and numc matrices with the same data and use
      cmp_dp_nc_matrix to compare the results
"""
class TestAdd(TestCase):
    def test_small_add(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_add(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(1000, 1000, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(1000, 1000, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_add(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(10000, 10000, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(10000, 10000, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)
    
    def test_type_error(self):
        try:
            nc.Matrix(3, 3) + 10
            self.assertTrue(False)
        except TypeError as e:
            print(e)
            pass
    
    def test_dimension_error(self):
        try:
            nc.Matrix(3, 3) + nc.Matrix(1, 1)
            self.assertTrue(False)
        except ValueError as e:
            print(e)
            pass
        

class TestSub(TestCase):
    def test_small_sub(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_sub(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(1000, 1000, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(1000, 1000, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_sub(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(10000, 10000, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(10000, 10000, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_type_error(self):
        try:
            nc.Matrix(5, 3) - 10
            self.assertTrue(False)
        except TypeError as e:
            print(e)
            pass
    
    def test_dimension_error(self):
        try:
            nc.Matrix(30, 30) - nc.Matrix(100, 1)
            self.assertTrue(False)
        except ValueError as e:
            print(e)
            pass


class TestAbs(TestCase):
    def test_small_abs(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_abs(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(550, 500, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_abs(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2100, 2000, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)


class TestNeg(TestCase):
    def test_small_neg(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_neg(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(500, 350, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_neg(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2000, 2500, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestMul(TestCase):
    def test_small_mul(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_mul(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(350, 1000, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(1000, 500, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_z_large_mul(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(3000, 3000, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(3000, 3000, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_type_error(self):
        try:
            nc.Matrix(30, 30) * 500
            self.assertTrue(False)
        except TypeError as e:
            print(e)
            pass
    
    def test_value_error(self):
        try:
            nc.Matrix(30, 30) * nc.Matrix(100, 1)
            self.assertTrue(False)
        except ValueError as e:
            print(e)
            pass

class TestPow(TestCase):
    def test_small_pow(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        is_correct, speed_up = compute([dp_mat, 3], [nc_mat, 3], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_pow(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(30, 30, seed=0)
        is_correct, speed_up = compute([dp_mat, 3], [nc_mat, 3], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_pow(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(200, 200, seed=0)
        is_correct, speed_up = compute([dp_mat, 4], [nc_mat, 4], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_power_of_one(self):
        dp_mat, nc_mat = rand_dp_nc_matrix(30, 30, seed=0)
        is_correct, speed_up = compute([dp_mat, 1], [nc_mat, 1], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_type_error(self):
        try:
            nc.Matrix(5, 5) ** "abc"
            self.assertTrue(False)
        except TypeError as e:
            print(e)
            pass
    
    def test_value_error1(self):
        try:
            nc.Matrix(4, 5) ** 10
            self.assertTrue(False)
        except ValueError as e:
            print(e)
            pass
    
    def test_value_error2(self):
        try:
            nc.Matrix(10, 10) ** -9
            self.assertTrue(False)
        except ValueError as e:
            print(e)
            pass


class TestGet(TestCase):
    def test_get(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        self.assertEqual(round(dp_mat[rand_row][rand_col], decimal_places),
            round(nc_mat[rand_row][rand_col], decimal_places))

    def test_type_error1(self):
        try:
            mat = nc.Matrix(4, 4)
            mat.get(1, 2, 3)
            self.assertTrue(False)
        except TypeError as e:
            print(e)
            pass

    def test_index_error2(self):
        try:
            mat = nc.Matrix(4, 4)
            mat.get(0, 4.9)
            self.assertTrue(False)
        except IndexError as e:
            print(e)
            pass
    
    def test_index_error3(self):
        try:
            mat = nc.Matrix(4, 4)
            mat.get(56.1, 0)
            self.assertTrue(False)
        except IndexError as e:
            print(e)
            pass

    def test_index_error1(self):
        try:
            mat = nc.Matrix(1000, 1000)
            mat.get(1001, 60)
            self.assertTrue(False)
        except IndexError as e:
            print(e)
            pass

    def test_index_error2(self):
        try:
            mat = nc.Matrix(1000, 1000)
            mat.get(10, 20000)
            self.assertTrue(False)
        except IndexError as e:
            print(e)
            pass
    
    def test_index_error3(self):
        try:
            mat = nc.Matrix(10, 1000)
            mat.get(200000, 50000)
            self.assertTrue(False)
        except IndexError as e:
            print(e)
            pass

class TestSet(TestCase):
    def test_set(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        self.assertEqual(round(dp_mat[rand_row][rand_col], decimal_places),
            round(nc_mat[rand_row][rand_col], decimal_places))
    
    def test_type_error1(self):
        try:
            mat = nc.Matrix(100, 100)
            mat.set(100, 5)
            self.assertTrue(False)
        except TypeError as e:
            print(e)
            pass
    
    def test_type_error2(self):
        try:
            mat = nc.Matrix(100, 100)
            mat.set(10, 5, 4, 47)
            self.assertTrue(False)
        except TypeError as e:
            print(e)
            pass
    
    def test_type_error3(self):
        try:
            mat = nc.Matrix(100, 100)
            mat.set(98, 0, "abc")
            self.assertTrue(False)
        except TypeError as e:
            print(e)
            pass

    def test_index_error1(self):
        try:
            mat = nc.Matrix(100, 100)
            mat.set(1000, 4, 4.0)
            self.assertTrue(False)
        except IndexError as e:
            print(e)
            pass

    def test_index_error2(self):
        try:
            mat = nc.Matrix(100, 100)
            mat.set(38, 400, 3430)
            self.assertTrue(False)
        except IndexError as e:
            print(e)
            pass

    def test_index_error3(self):
        try:
            mat = nc.Matrix(100, 100)
            mat.set(380, 4000, 420)
            self.assertTrue(False)
        except IndexError as e:
            print(e)
            pass

class TestShape(TestCase):
    def test_shape(self):
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        self.assertTrue(dp_mat.shape == nc_mat.shape)
    
