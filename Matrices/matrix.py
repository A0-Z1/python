class Matrix:
    '''
    Creates a R-style matrix object
    '''

    def __init__(self, mylist, dim):
        '''
        initialize the object
        '''

        # each nested list is a column
        self.dim = dim
        self.matrix = self._matrix_creation(mylist)
        self.vector = self._flatten()

    def __iter__(self):
        '''
        Implement iterable behaviour
        '''

        return iter(self.matrix)


    def _matrix_creation(self, mylist):
        '''
        creates the matrix. If dimension
        tuple doesn't coincide with list length,
        recycle the elements of the list in true
        R fashion
        '''

        matrix = []
        counter = 0
        # iterate for every row
        for row in range(self.dim[0]):
            row_list = []
            # iterate for every column
            for column in range(self.dim[1]):
                row_list.append(mylist[counter])
                # if it reaches end of list, start from beginning
                counter = (counter + 1) % len(mylist)

            matrix.append(row_list)

        return matrix

    def _flatten(self):
        '''
        flatten the matrix
        '''

        vector = []
        for row in range(self.dim[0]):
            for column in range(self.dim[1]):
                vector.append(self.matrix[row][column])

        return vector

    def flat(self):
        '''
        return flat vector
        '''

        return self.vector


    def __eq__(self, other):
        '''
        Defines behaviour for equality operator ==
        '''

        # check dimensionality
        if self.dim != other.dim:
            return False
        else:
            # just check if all the elements are the same
            for index in range(len(self.vector)):
                if self.vector[index] != other.vector[index]:
                    return False

            else:
                return True


    ### Unary Operators and functions

    def __neg__(self):
        '''
        Implements behaviour for negation
        '''

        # negate all elements
        neg_list = [-el for el in self.vector]
        neg_matrix = Matrix(neg_list, dim=self.dim)

        return neg_matrix

    def __pos__(self):
        '''
        Implements behaviour for unary positive
        '''

        pos_matrix = Matrix(self.vector, dim=self.dim)

        return pos_matrix

    def __abs__(self):
        '''
        Implements behaviour for built in `abs` function
        '''

        abs_list = [abs(el) for el in self.vector]
        abs_matrix = Matrix(abs_list, dim = self.dim)

        return abs_matrix

    def __round__(self, n):
        '''
        Implements behaviour for built in `round` function
        '''

        round_list = [round(el, n) for el in self.vector]
        round_matrix = Matrix(round_list, dim = self.dim)

        return round_matrix

    def T(self):
        '''
        Transpose the matrix
        '''

        vector = [self.matrix[i][j] for j in range(self.dim[1]) for i in range(self.dim[0])]
        dim1 = self.dim[1]
        dim2 = self.dim[0]

        transposed_mat = Matrix(vector, dim = (dim1, dim2))

        return transposed_mat

    ### Normal arithmetic operators

    def _check_dim(self, other):
        '''
        check that dimensions coincide
        '''
        if self.dim != other.dim:
            raise IndexNotMatched

    def _scalar_operation(self, other):
        '''
        Allow broadcasting
        '''

        if isinstance(other, int) or isinstance(other, float):
            return Matrix([other], dim = self.dim)
        else:
            return other

    def __add__(self, other):
        '''
        Implement addition
        '''

        other = self._scalar_operation(other)
        # check dimensions
        self._check_dim(other)

        add_list = [sum(pair) for pair in zip(self.vector, other.vector)]
        add_matrix = Matrix(add_list, dim = self.dim)

        return add_matrix

    def __sub__(self, other):
        '''
        Implement subtraction
        '''

        other = self._scalar_operation(other)
        # check dimensions
        self._check_dim(other)

        sub_list = [pair[0] - pair[1] for pair in zip(self.vector, other.vector)]
        sub_matrix = Matrix(sub_list, dim = self.dim)

        return sub_matrix

    def __mul__(self, other):
        '''
        Implement multiplication
        '''

        other = self._scalar_operation(other)
        # check dimensions
        self._check_dim(other)

        mult_list = [pair[0] * pair[1] for pair in zip(self.vector, other.vector)]
        mult_matrix = Matrix(mult_list, dim = self.dim)

        return mult_matrix

    def __floordiv__(self, other):
        '''
        Implement integer division using //
        '''

        other = self._scalar_operation(other)
        # check dimensions
        self._check_dim(other)

        floordiv_list = [pair[0] // pair[1] for pair in zip(self.vector, other.vector)]
        floordiv_matrix = Matrix(floordiv_list, dim = self.dim)

        return floordiv_matrix

    def __truediv__(self, other):
        '''
        Implement division using the / operator
        '''

        other = self._scalar_operation(other)
        # check dimensions
        self._check_dim(other)

        div_list = [pair[0] / pair[1] for pair in zip(self.vector, other.vector)]
        div_matrix = Matrix(div_list, dim = self.dim)

        return div_matrix

    def __mod__(self, other):
        '''
        implement modulo using the % operator
        '''

        other = self._scalar_operation(other)
        # check dimensions
        self._check_dim(other)

        mod_list = [pair[0] % pair[1] for pair in zip(self.vector, other.vector)]
        mod_matrix = Matrix(mod_list, dim = self.dim)

        return mod_matrix

    def __pow__(self, power):
        '''
        Implement behavior for exponent using **
        '''

        if self.dim[0] != self.dim[1]:
            raise IndexNotMatched

        # multiply the matrix with itself
        pow_mat = self
        for i in range(power-1):
            pow_mat = pow_mat @ self

        return pow_mat

    def power(self, other):
        '''
        Implement component-wise exponent
        '''

        other = self._scalar_operation(other)
        # check dimensions
        self._check_dim(other)

        pow_list = [pair[0] ** pair[1] for pair in zip(self.vector, other.vector)]
        pow_matrix = Matrix(pow_list, dim = self.dim)

        return pow_matrix


    def __matmul__(self, other):
        '''
        Implement matrix multiplication
        (@ operator)
        '''

        # check dimensions:
        if self.dim[1] != other.dim[0]:
            raise IndexNotMatched

        dim = (self.dim[0], other.dim[1])
        matmul_list = []
        for row_A in range(self.dim[0]):
            for column_B in range(other.dim[1]):
                el = 0
                for column_A in range(self.dim[1]):
                    el += self.matrix[row_A][column_A] * other.matrix[column_A][column_B]
                matmul_list.append(el)

        matmul_matrix = Matrix(matmul_list, dim = dim)

        return matmul_matrix


    ### Representation
    
    def __str__(self):
        '''
        Define behaviour for when str() is called
        on an instance of the class
        Basically defines what `print` will output
        '''

        pretty_mat = "\n"
        for row in range(self.dim[0]):
            for col in range(self.dim[1]):
                pretty_mat += f"{self.matrix[row][col]:10.1f} "

            pretty_mat += "\n\n"

        return pretty_mat

    def __repr__(self):
        '''
        Define behaviour for when repr() is called on an
        instance of the class
        Basically defines what calling the object will output
        '''

        return f"matrix_obj{self.matrix}"

    ### Sequences

    def reshape(self, new_dim):
        '''
        reshape matrix
        '''

        new_list = self.vector
        new_matrix = Matrix(new_list, dim = new_dim)

        return new_matrix

    def __len__(self):
        '''
        Return length of matrix
        '''

        return len(self.vector)

    def __getitem__(self, key):
        '''
        Define behaviour for when an item is assigned to,
        with notation self[key]
        '''

        # slice the rows
        # make sure to return
        # a nested list in the same
        # form of a normal matrix
        if isinstance(key[0], int):
            rows = [self.matrix[key[0]]]
        else:
            rows = self.matrix[key[0]]

        # determine number of rows
        dim1 = len(rows)

        # slice the columns. As before
        # return nested lists in the same form
        # of a normal matrix
        for count, el in enumerate(rows):
            if isinstance(key[1], int):
                rows[count] = [el[key[1]]]
            else:
                rows[count] = el[key[1]]

        # determine no of columns
        dim2 = len(rows[0])

        # build the vector
        vector = []
        for row in rows:
            vector.extend(row)

        # build the matrix object
        sliced_matrix = Matrix(vector, dim = (dim1, dim2))

        return sliced_matrix

    def __setitem__(self, key, value):
        '''
        Define behaviour for when an item is assigned to,
        with notation self[key] = value
        '''
        pass

    def item(self):
        '''
        if matrix object is only a value,
        turn it into float
        '''

        if self.dim == (1, 1):
            return self.matrix[0][0]
        else:
            raise TypeError

    @staticmethod
    def diag(mylist):
        '''
        Creates a diagonal matrix with
        values provided as diagonal
        '''

        dim = len(mylist)
        vector = [0] * (dim ** 2)
        for i in range(dim):
            vector[i * (dim + 1)] = mylist[i]

        matrix = Matrix(vector, dim = (dim, dim))

        return matrix

    @staticmethod
    def I(dim):
        '''
        Creates the identity matrix
        '''

        mylist = [1] * dim

        return diag(mylist)




class IndexNotMatched(Exception):
    pass
