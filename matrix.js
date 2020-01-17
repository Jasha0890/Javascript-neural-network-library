/*
* Matrix library to build a neural network to
* feedforward and backpropagate the data
* this library has all the matrix operations
* needed and used in the neural network library
*/
class Matrix {
  constructor(rows, columns) {
    this.rows = rows;
    this.columns = columns;
    this.data = [];

    for (let i = 0; i < this.rows; i++) {
      this.data[i] = [];
      for (let j = 0; j < this.columns; j++) {
        this.data[i][j] = 0;
      }
    }
  }
  //converts an array into a matrix
  static fromArray(arr) {
    let m = new Matrix(arr.length, 1);
    for (let i = 0; i < arr.length; i++) {
      m.data[i][0] = arr[i];
    }
    return m;
  }

  static subtract(a, b) {
    // Return a new Matrix a-b
    let result = new Matrix(a.rows, a.columns);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.columns; j++) {
        result.data[i][j] = a.data[i][j] - b.data[i][j];
      }
    }
    return result;
  }

  toArray() {
    let arr = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }

  randomize() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        this.data[i][j] = Math.random() * 2 - 1;
      }
    }
  }

  add(n) {
    if (n instanceof Matrix) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.data[i][j] += n.data[i][j];
        }
      }
    } else {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.data[i][j] += n;
        }
      }
    }
  }

  static transpose(matrix) {
    let result = new Matrix(matrix.columns, matrix.rows);
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.columns; j++) {
        result.data[j][i] = matrix.data[i][j];
      }
    }
    return result;
  }

  static multiply(a, b) {
    // Matrix product
    if (a.columns !== b.rows) {
      console.log('Columns of A must match rows of B.')
      return undefined;
    }
    let result = new Matrix(a.rows, b.columns);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.columns; j++) {
        // Dot product of values in col
        let sum = 0;
        for (let k = 0; k < a.columns; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    return result;
  }

  multiply(n) {
    if (n instanceof Matrix) {
      // hadamard product
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.data[i][j] *= n.data[i][j];
        }
      }
    } else {
      // Scalar product
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.columns; j++) {
          this.data[i][j] *= n;
        }
      }
    }
  }

  map(func) {
    // Apply a function to every element of matrix
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        let val = this.data[i][j];
        this.data[i][j] = func(val);
      }
    }
  }

  static map(matrix, func) {
    let result = new Matrix(matrix.rows, matrix.columns);
    // Apply a function to every element of matrix
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.columns; j++) {
        let val = matrix.data[i][j];
        result.data[i][j] = func(val);
      }
    }
    return result;
  }

  print() {
    console.table(this.data);
  }
}


if (typeof module !== 'undefined') {
  module.exports = Matrix;
}