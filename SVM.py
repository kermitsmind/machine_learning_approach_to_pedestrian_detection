import numpy as np
import random as rnd


# based on: https://github.com/akuchotrani/SupportVectorMacine/blob/master/SVM.py
class SVM:
    """
    Class responsible for performing Support Vector Machine algorithm calculation
    """

    def __init__(self, max_iter=10000, kernel_type="linear", C=1.0, epsilon=0.001):
        """
        Method responsible for initializing the class with parsed parameters.

        :param: max_iter: maximal number of the algorithm iterations
        :param: kernel_type: type of a kernel
        :param: C: regularization parameter
        :param: epsilon: convergence criterion
        """

        self.kernels = {
            "linear": self.kernel_linear,
            "quadratic": self.kernel_quadratic,
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon

    def calculate_b(self, X, Y, w):
        """
        Method responsible for calculating b value

        :param: X: ...
        :param: Y: ...
        :param: w: ...

        :return ...
        """

        b_tmp = Y - np.dot(w.T, X.T)

        return np.mean(b_tmp)

    def calculate_w(self, alpha, Y, X):
        """
        Method responsible for calculating w value

        :param: alpha: ...
        :param: Y: ...
        :param: X: ...

        :return ...
        """

        return np.dot(alpha * Y, X)

    def h(self, X, w, b):
        """
        Method

        :param: X: ...
        :param: w: ...
        :param: b: ...

        :return ...
        """

        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    def E(self, x_k, y_k, w, b):
        """
        Method responsible for calculating the prediction error

        :param: x_k: ...
        :param: y_k: ...
        :param: w: ...
        :param: b: ...

        :return ...
        """

        return self.h(x_k, w, b) - y_k

    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        """
        Method ...

        :param: C: ...
        :param: alpha_prime_j: ...
        :param: alpha_prime_i: ...
        :param: y_j: ...
        :param: y_i: ...

        :return ...
        """

        if y_i != y_j:

            return (
                max(0, alpha_prime_j - alpha_prime_i),
                min(C, C - alpha_prime_i + alpha_prime_j),
            )
        else:

            return (
                max(0, alpha_prime_i + alpha_prime_j - C),
                min(C, alpha_prime_i + alpha_prime_j),
            )

    def get_rnd_int(self, a, b, z):
        """
        Method ...

        :param: a: ...
        :param: b: ...
        :param: z: ...

        :return ...
        """

        i = z
        cnt = 0
        while i == z and cnt < 1000:
            i = rnd.randint(a, b)
            cnt = cnt + 1

        return i

    def kernel_linear(self, x1, x2):
        """
        Method defining linear kernelel type

        :param: x1: ...
        :param: x2: ...

        :return ...
        """

        return np.dot(x1, x2.T)

    def kernel_quadratic(self, x1, x2):
        """
        Method defining quadratic kernel type

        :param: x1: ...
        :param: x2: ...

        :return ...
        """

        return np.dot(x1, x2.T) ** 2

    def performFit(self, X, Y):
        """
        Method ...

        :param: X: x train data (feature data)
        :param: Y: y train data (labels)

        :return ...
        """

        # n : number of samples (100)
        n = X.shape[0]
        # we have alpha per sample of training set. Initially set to zeros
        alpha = np.zeros((n))
        # pick the kernel user selected
        kernel = self.kernels[self.kernel_type]
        iteration = 0
        while True:
            iteration += 1
            # saving the copy of alpha from previous iteration
            alpha_prev = np.copy(alpha)
            # going through all the samples in one iteration
            for j in range(0, n):
                # selcting random sample index where i is not equal to j
                i = self.get_rnd_int(0, n - 1, j)  # Get random int i~=j
                x_i = X[i, :]
                x_j = X[j, :]
                y_i = Y[i]
                y_j = Y[j]
                k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)
                if k_ij == 0:
                    continue
                # select alpha of i and j from the alpha array to calculate L and H
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                (L, H) = self.compute_L_H(
                    self.C, alpha_prime_j, alpha_prime_i, y_j, y_i
                )
                # Compute model parameters
                self.w = self.calculate_w(alpha, Y, X)
                self.b = self.calculate_b(X, Y, self.w)
                # Compute E_i, E_j
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)
                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j)) / k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)
                alpha[i] = alpha_prime_i + y_i * y_j * (alpha_prime_j - alpha[j])
            # Terminating condition: reacing convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:

                break

            # Terminating condition: Reaching max iterations
            if iteration >= self.max_iter:
                print(
                    "Iteration number exceeded the max of %d iterations"
                    % (self.max_iter)
                )

                return

        # Compute final model parameters
        self.b = self.calculate_b(X, Y, self.w)
        if self.kernel_type == "linear":
            self.w = self.calculate_w(alpha, Y, X)

    def makePrediction(self, X):
        """
        Method ...

        :param: X: ...

        :return ...
        """

        return self.h(X, self.w, self.b)

    def printInfo(self):
        """
        Method printing out the parameters of SVM
        """

        print("############ PRINT SVM INFO ################")
        print("C:", self.C)
        print("max_iter:", self.max_iter)
        print("epsilon:", self.epsilon)
        print("kernel_type:", self.kernel_type)


def calculateAccuracy(y, y_hat):
    """
    Function responsible for calculating accuracy of the prediction

    :param: y: actual data
    :param: y_hat: predicted data

    :return prediction accuracy
    """

    correct_counter = 0
    for i in range(0, len(y)):
        if y[i] == -1 and y_hat[i] == -1:
            correct_counter = correct_counter + 1
        if y[i] == 1 and y_hat[i] == 1:
            correct_counter = correct_counter + 1

    return correct_counter / len(y)
