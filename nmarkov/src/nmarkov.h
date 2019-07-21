#include <string>
#include <cstdio>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//#include <pybind11/iostream.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include <marlib.h>

namespace py = pybind11;
using ipyarray = py::array_t<int32_t, py::array::f_style>;
using dpyarray = py::array_t<double, py::array::f_style>;

inline
int array_size(const dpyarray& x) {
    py::buffer_info info = const_cast<dpyarray&>(x).request();
    int size = 1;
    for (int i=0; i<info.ndim; i++) {
        size *= info.shape[i];
    }
    return size;
}

namespace marlib {

  template <>
  struct vector_traits<dpyarray> {
    static int size(const dpyarray& v) { return array_size(v); }
    static const double* value(const dpyarray& v) { return v.data(); }
    static double* value(dpyarray& v) { return v.mutable_data(); }
    static int inc(const dpyarray& v) { return 1; }
  };

  template <>
  struct dense_matrix_traits<dpyarray> {
    static int nrow(const dpyarray& m) { return const_cast<dpyarray&>(m).request().shape[0]; }
    static int ncol(const dpyarray& m) { return const_cast<dpyarray&>(m).request().shape[1]; }
    static const double* value(const dpyarray& m) { return m.data(); }
    static double* value(dpyarray& m) { return m.mutable_data(); }
    static int ld(const dpyarray& m) { return const_cast<dpyarray&>(m).request().shape[0]; }
  };

  template <>
  struct vector_traits<py::object> {
    static int size(const py::object& v) { return py::reinterpret_borrow<py::int_>(v.attr("nnz")); }
    static const double* value(const py::object& v) { return py::reinterpret_borrow<dpyarray>(v.attr("data")).data(); }
    static double* value(py::object& v) { return py::reinterpret_borrow<dpyarray>(v.attr("data")).mutable_data(); }
    static int inc(const py::object& v) { return 1; }
  };

  template <>
  struct csr_matrix_traits<py::object> {
    static int nrow(const py::object& m) { return py::reinterpret_borrow<py::int_>(py::reinterpret_borrow<py::tuple>(m.attr("shape"))[0]); }
    static int ncol(const py::object& m) { return py::reinterpret_borrow<py::int_>(py::reinterpret_borrow<py::tuple>(m.attr("shape"))[1]); }
    static int nnz(const py::object& m) { return py::reinterpret_borrow<py::int_>(m.attr("nnz")); }
    static int base(const py::object& m) { return 0; }
    static double* value(py::object& m) { return py::reinterpret_borrow<dpyarray>(m.attr("data")).mutable_data(); }
    static const double* value(const py::object& m) { return py::reinterpret_borrow<dpyarray>(m.attr("data")).data(); }
    static const int* rowptr(const py::object& m) { return py::reinterpret_borrow<ipyarray>(m.attr("indptr")).data(); }
    static const int* colind(const py::object& m) { return py::reinterpret_borrow<ipyarray>(m.attr("indices")).data(); }
  };

  template <>
  struct csc_matrix_traits<py::object> {
    static int nrow(const py::object& m) { return py::reinterpret_borrow<py::int_>(py::reinterpret_borrow<py::tuple>(m.attr("shape"))[0]); }
    static int ncol(const py::object& m) { return py::reinterpret_borrow<py::int_>(py::reinterpret_borrow<py::tuple>(m.attr("shape"))[1]); }
    static int nnz(const py::object& m) { return py::reinterpret_borrow<py::int_>(m.attr("nnz")); }
    static int base(const py::object& m) { return 0; }
    static double* value(py::object& m) { return py::reinterpret_borrow<dpyarray>(m.attr("data")).mutable_data(); }
    static const double* value(const py::object& m) { return py::reinterpret_borrow<dpyarray>(m.attr("data")).data(); }
    static const int* colptr(const py::object& m) { return py::reinterpret_borrow<ipyarray>(m.attr("indptr")).data(); }
    static const int* rowind(const py::object& m) { return py::reinterpret_borrow<ipyarray>(m.attr("indices")).data(); }
  };

  template <>
  struct coo_matrix_traits<py::object> {
    static int nrow(const py::object& m) { return py::reinterpret_borrow<py::int_>(py::reinterpret_borrow<py::tuple>(m.attr("shape"))[0]); }
    static int ncol(const py::object& m) { return py::reinterpret_borrow<py::int_>(py::reinterpret_borrow<py::tuple>(m.attr("shape"))[1]); }
    static int nnz(const py::object& m) { return py::reinterpret_borrow<py::int_>(m.attr("nnz")); }
    static int base(const py::object& m) { return 0; }
    static double* value(py::object& m) { return py::reinterpret_borrow<dpyarray>(m.attr("data")).mutable_data(); }
    static const double* value(const py::object& m) { return py::reinterpret_borrow<dpyarray>(m.attr("data")).data(); }
    static const int* rowind(const py::object& m) { return py::reinterpret_borrow<ipyarray>(m.attr("row")).data(); }
    static const int* colind(const py::object& m) { return py::reinterpret_borrow<ipyarray>(m.attr("col")).data(); }
  };

  template <typename T1, typename TypeT>
  T1 clone(const T1& x, TypeT);

  template <>
  dpyarray clone(const dpyarray& x, ArrayT) {
    using traits = vector_traits<dpyarray>;
    const int n = traits::size(x);
    dpyarray y(n);
    dcopy(x, y);
    return y;
  }

  template <>
  dpyarray clone(const dpyarray& x, DenseMatrixT) {
    dpyarray y(const_cast<dpyarray&>(x).request().shape);
    dcopy(x, y);
    return y;
  }

  template <>
  py::object clone(const py::object& x, CSRMatrixT) {
    return x.attr("copy")();
  }

  template <>
  py::object clone(const py::object& x, CSCMatrixT) {
    return x.attr("copy")();
  }

  template <>
  py::object clone(const py::object& x, COOMatrixT) {
    return x.attr("copy")();
  }
}

template <typename ... Args>
std::string format(const std::string& fmt, Args ... args ) {
  size_t len = std::snprintf( nullptr, 0, fmt.c_str(), args ... );
  std::vector<char> buf(len + 1);
  std::snprintf(&buf[0], len + 1, fmt.c_str(), args ... );
  return std::string(&buf[0], &buf[0] + len);
}
