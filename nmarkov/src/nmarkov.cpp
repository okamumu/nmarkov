#include "nmarkov.h"
#include <tuple>

namespace nmarkov {

  using Params = marlib::marlib_params;

  template <typename T1, typename MatT>
  dpyarray ctmc_st_gth(const T1& Q) {
    const int n = marlib::nrow(Q, MatT());
    dpyarray x(n);
    marlib::ctmc_st_gth(Q, x, MatT());
    return x;
  }

  template <typename T1, typename MatT>
  dpyarray ctmc_st_power(const T1& Q, const dpyarray& x0, Params& params) {
    const int n = marlib::nrow(Q, MatT());
    T1 P = clone(Q, MatT());
    dpyarray x(n);
    marlib::dcopy(x0, x);
    marlib::ctmc_st_power(P, x, params, [](Params){}, MatT());
    return x;
  }

  template <typename T1, typename MatT>
  dpyarray ctmc_st_gs(const T1& Q, const dpyarray& x0, Params& params) {
    const int n = marlib::nrow(Q, MatT());
    dpyarray x(n);
    marlib::dcopy(x0, x);
    marlib::ctmc_st_gs(Q, x, params, [](Params){}, MatT());
    return x;
  }

  template <typename T1, typename MatT>
  dpyarray ctmc_stsen_gs(const T1& Q, const dpyarray& x0, const dpyarray& b, const dpyarray& pis, Params& params) {
    const int n = marlib::nrow(Q, MatT());
    dpyarray x(n);
    marlib::dcopy(x0, x);
    marlib::ctmc_stsen_gs(Q, x, b, pis, params, [](Params){}, MatT());
    return x;
  }

  template <typename T1, typename TR, typename MatT, typename VecT>
  dpyarray ctmc_mexp(TR, const T1& Q, const dpyarray& x, double t, Params& params, MatT, VecT) {
    T1 P = clone(Q, MatT());
    dpyarray y = clone(x, VecT());
    marlib::mexp_func(TR(), P, x, y, t, params,
              [](Params& params){
                throw std::runtime_error(format("Time interval is too large: right = %d (rmax: %d).", params.r, params.rmax));
                },
              MatT(), VecT());
    return y;
  }

  template <typename T1, typename MatT>
  dpyarray ctmc_mexp(const T1& Q, const dpyarray& x, double t, bool trans, Params& params) {
    int ndim = const_cast<dpyarray&>(x).request().ndim;
    if (ndim == 1 && trans) {
      return ctmc_mexp(marlib::TRANS(), Q, x, t, params, MatT(), marlib::ArrayT());
    } else if (ndim == 1 && !trans) {
      return ctmc_mexp(marlib::NOTRANS(), Q, x, t, params, MatT(), marlib::ArrayT());
    } else if (ndim == 2 && trans) {
      return ctmc_mexp(marlib::TRANS(), Q, x, t, params, MatT(), marlib::DenseMatrixT());
    } else if (ndim == 2 && !trans) {
      return ctmc_mexp(marlib::NOTRANS(), Q, x, t, params, MatT(), marlib::DenseMatrixT());
    } else {
      throw std::runtime_error("The argument x is neither vector nor dense matrix.");
    }
  }

  template <typename T1, typename TR, typename MatT, typename VecT>
  std::tuple<dpyarray,dpyarray> ctmc_mexpint(TR, const T1& Q, const dpyarray& x, const dpyarray& cx, double t, Params& params, MatT, VecT) {
    T1 P = clone(Q, MatT());
    dpyarray y = clone(x, VecT());
    dpyarray cy = clone(cx, VecT());
    marlib::mexpint_func(TR(), P, x, y, cy, t, params,
              [](Params& params){
                throw std::runtime_error(format("Time interval is too large: right = %d (rmax: %d).", params.r, params.rmax));
                },
              MatT(), VecT());
    return std::make_tuple(y, cy);
  }

  template <typename T1, typename MatT>
  std::tuple<dpyarray,dpyarray> ctmc_mexpint(const T1& Q, const dpyarray& x, const dpyarray& cx, double t, bool trans, Params& params) {
    int ndim = const_cast<dpyarray&>(x).request().ndim;
    if (ndim == 1 && trans) {
      return ctmc_mexpint(marlib::TRANS(), Q, x, cx, t, params, MatT(), marlib::ArrayT());
    } else if (ndim == 1 && !trans) {
      return ctmc_mexpint(marlib::NOTRANS(), Q, x, cx, t, params, MatT(), marlib::ArrayT());
    } else if (ndim == 2 && trans) {
      return ctmc_mexpint(marlib::TRANS(), Q, x, cx, t, params, MatT(), marlib::DenseMatrixT());
    } else if (ndim == 2 && !trans) {
      return ctmc_mexpint(marlib::NOTRANS(), Q, x, cx, t, params, MatT(), marlib::DenseMatrixT());
    } else {
      throw std::runtime_error("The argument x is neither vector nor dense matrix.");
    }
  }

  template <typename T1, typename TR, typename MatT, typename VecT>
  std::tuple<dpyarray,dpyarray> ctmc_tran(TR, const T1& Q, const dpyarray& x, const dpyarray& cx, const dpyarray& t, Params& params, MatT, VecT) {
    T1 P = clone(Q, MatT());
    dpyarray y = clone(x, VecT());
    dpyarray cy = clone(cx, VecT());
    const int m = array_size(t);
    const int s = array_size(x);
    dpyarray res_x(m*s);
    dpyarray res_cx(m*s);
    marlib::ctmc_tran(TR(), P, y, cy, t, res_x, res_cx, params,
              [](Params& params){
                throw std::runtime_error(format("Time interval is too large: right = %d (rmax: %d).", params.r, params.rmax));
                },
              [](Params){},
              MatT(), VecT());
    return std::make_tuple(res_x, res_cx);
  }

  template <typename T1, typename MatT>
  std::tuple<dpyarray,dpyarray> ctmc_tran(const T1& Q, const dpyarray& x, const dpyarray& cx, const dpyarray& t, bool trans, Params& params) {
    int ndim = const_cast<dpyarray&>(x).request().ndim;
    if (ndim == 1 && trans) {
      return ctmc_tran(marlib::TRANS(), Q, x, cx, t, params, MatT(), marlib::ArrayT());
    } else if (ndim == 1 && !trans) {
      return ctmc_tran(marlib::NOTRANS(), Q, x, cx, t, params, MatT(), marlib::ArrayT());
    } else if (ndim == 2 && trans) {
      return ctmc_tran(marlib::TRANS(), Q, x, cx, t, params, MatT(), marlib::DenseMatrixT());
    } else if (ndim == 2 && !trans) {
      return ctmc_tran(marlib::NOTRANS(), Q, x, cx, t, params, MatT(), marlib::DenseMatrixT());
    } else {
      throw std::runtime_error("The argument x is neither vector nor dense matrix.");
    }
  }

  template <typename T1, typename TR, typename MatT, typename VecT, typename VecT2>
  std::tuple<dpyarray,dpyarray,dpyarray,dpyarray> ctmc_tran_rwd(TR, const T1& Q, const dpyarray& x, const dpyarray& cx, const dpyarray& rwd, const dpyarray& t, Params& params, MatT, VecT, VecT2) {
    T1 P = clone(Q, MatT());
    dpyarray y = clone(x, VecT());
    dpyarray cy = clone(cx, VecT());
    const int m = array_size(t);
    const int s = marlib::get_size(x, rwd, VecT(), VecT2());
    dpyarray res_irwd(m*s);
    dpyarray res_crwd(m*s);
    marlib::ctmc_tran_rwd(TR(), P, y, cy, rwd, t, res_irwd, res_crwd, params,
              [](Params& params){
                throw std::runtime_error(format("Time interval is too large: right = %d (rmax: %d).", params.r, params.rmax));
                },
              [](Params){},
              MatT(), VecT(), VecT2());
    return std::make_tuple(y, cy, res_irwd, res_crwd);
  }

  template <typename T1, typename MatT>
  std::tuple<dpyarray,dpyarray,dpyarray,dpyarray> ctmc_tran_rwd(const T1& Q, const dpyarray& x, const dpyarray& cx, const dpyarray& rwd, const dpyarray& t, bool trans, Params& params) {
    int ndim = const_cast<dpyarray&>(x).request().ndim;
    int ndimr = const_cast<dpyarray&>(rwd).request().ndim;
    if (ndim == 1 && ndimr == 1 && trans) {
      return ctmc_tran_rwd(marlib::TRANS(), Q, x, cx, rwd, t, params, MatT(), marlib::ArrayT(), marlib::ArrayT());
    } else if (ndim == 1 && ndimr == 1 && !trans) {
      return ctmc_tran_rwd(marlib::NOTRANS(), Q, x, cx, rwd, t, params, MatT(), marlib::ArrayT(), marlib::ArrayT());
    } else if (ndim == 1 && ndimr == 2 && trans) {
      return ctmc_tran_rwd(marlib::TRANS(), Q, x, cx, rwd, t, params, MatT(), marlib::ArrayT(), marlib::DenseMatrixT());
    } else if (ndim == 1 && ndimr == 2 && !trans) {
      return ctmc_tran_rwd(marlib::NOTRANS(), Q, x, cx, rwd, t, params, MatT(), marlib::ArrayT(), marlib::DenseMatrixT());
    } else if (ndim == 2 && ndimr == 2 && trans) {
      return ctmc_tran_rwd(marlib::TRANS(), Q, x, cx, rwd, t, params, MatT(), marlib::DenseMatrixT(), marlib::DenseMatrixT());
    } else if (ndim == 2 && ndimr == 2 && !trans) {
      return ctmc_tran_rwd(marlib::NOTRANS(), Q, x, cx, rwd, t, params, MatT(), marlib::DenseMatrixT(), marlib::DenseMatrixT());
    } else if (ndim == 2 && ndimr == 1) {
      throw std::runtime_error("The argument rwd should be a matrix when x is a matrix.");
    } else {
      throw std::runtime_error("The argument x is neither vector nor dense matrix.");
    }
  }
}

PYBIND11_MODULE(_nmarkov, m) {
  m.doc() = "Numerical computation for Markov chains";

  py::class_<marlib::marlib_params>(m, "Params")
    .def(py::init<>())
    .def_readwrite("ufact", &marlib::marlib_params::ufact)
    .def_readwrite("eps", &marlib::marlib_params::eps)
    .def_readwrite("rmax", &marlib::marlib_params::rmax)
    .def_readwrite("rtol", &marlib::marlib_params::rtol)
    .def_readwrite("steps", &marlib::marlib_params::steps)
    .def_readwrite("maxiter", &marlib::marlib_params::maxiter)
    .def_readwrite("iter", &marlib::marlib_params::iter)
    .def_readwrite("info", &marlib::marlib_params::info)
    .def_readwrite("r", &marlib::marlib_params::r)
    .def_readwrite("rerror", &marlib::marlib_params::rerror)
    ;

  m.def("ctmc_st_gth_dense",
    nmarkov::ctmc_st_gth<dpyarray,marlib::DenseMatrixT>,
    py::arg("Q"),
    "Compute the steady-state vector with GTH");

  m.def("ctmc_st_gth_sparse",
    [](const py::object& Q) {
      std::string mattype = py::reinterpret_borrow<py::str>(Q.attr("format"));
      if (mattype == "csr") {
        return nmarkov::ctmc_st_gth<py::object,marlib::CSRMatrixT>(Q);
      } else if (mattype == "csc") {
        return nmarkov::ctmc_st_gth<py::object,marlib::CSCMatrixT>(Q);
      } else if (mattype == "coo") {
        return nmarkov::ctmc_st_gth<py::object,marlib::COOMatrixT>(Q);
      } else {
        throw std::runtime_error("Q is either csr, csc and coo.");
      }
    },
    py::arg("Q"),
    "Compute the steady-state vector with GTH");

  m.def("ctmc_st_power_dense",
    nmarkov::ctmc_st_power<dpyarray,marlib::DenseMatrixT>,
    py::arg("Q"), py::arg("x0"), py::arg("params"),
    "Compute the steady-state vector with GTH");

  m.def("ctmc_st_power_sparse",
    [](const py::object& Q, const dpyarray& x0, marlib::marlib_params& params) {
      std::string mattype = py::reinterpret_borrow<py::str>(Q.attr("format"));
      if (mattype == "csr") {
        return nmarkov::ctmc_st_power<py::object,marlib::CSRMatrixT>(Q, x0, params);
      } else if (mattype == "csc") {
        return nmarkov::ctmc_st_power<py::object,marlib::CSCMatrixT>(Q, x0, params);
      } else if (mattype == "coo") {
        return nmarkov::ctmc_st_power<py::object,marlib::COOMatrixT>(Q, x0, params);
      } else {
        throw std::runtime_error("Q is either csr, csc and coo.");
      }
    },
    py::arg("Q"), py::arg("x0"), py::arg("params"),
    "Compute the steady-state vector with GS");

  m.def("ctmc_st_gs_dense",
    nmarkov::ctmc_st_gs<dpyarray,marlib::DenseMatrixT>,
    py::arg("Q"), py::arg("x0"), py::arg("params"),
    "Compute the steady-state vector with GS");

  m.def("ctmc_st_gs_sparse",
    [](const py::object& Q, const dpyarray& x0, marlib::marlib_params& params) {
      std::string mattype = py::reinterpret_borrow<py::str>(Q.attr("format"));
      if (mattype == "csc") {
        return nmarkov::ctmc_st_gs<py::object,marlib::CSCMatrixT>(Q, x0, params);
      } else {
        throw std::runtime_error("Q should be csc.");
      }
    },
    py::arg("Q"), py::arg("x0"), py::arg("params"),
    "Compute the steady-state vector with GS");

  m.def("ctmc_stsen_gs_dense",
    nmarkov::ctmc_stsen_gs<dpyarray,marlib::DenseMatrixT>,
    py::arg("Q"), py::arg("x0"), py::arg("b"), py::arg("pis"), py::arg("params"),
    "Compute the sensitivity of steady-state vector with GS");

  m.def("ctmc_stsen_gs_sparse",
    [](const py::object& Q, const dpyarray& x0, const dpyarray& b, const dpyarray& pis, marlib::marlib_params& params) {
      std::string mattype = py::reinterpret_borrow<py::str>(Q.attr("format"));
      if (mattype == "csc") {
        return nmarkov::ctmc_stsen_gs<py::object,marlib::CSCMatrixT>(Q, x0, b, pis, params);
      } else {
        throw std::runtime_error("Q should be csc.");
      }
    },
    py::arg("Q"), py::arg("x0"), py::arg("b"), py::arg("pis"), py::arg("params"),
    "Compute the sensitivity of steady-state vector with GS");

  m.def("ctmc_mexp_dense",
    nmarkov::ctmc_mexp<dpyarray,marlib::DenseMatrixT>,
    py::arg("Q"), py::arg("x"), py::arg("t"), py::arg("trans"), py::arg("params"),
    "Compute the matrix exponential with uniformization");

  m.def("ctmc_mexp_sparse",
    [](const py::object& Q, const dpyarray& x, double t, bool trans, marlib::marlib_params& params) {
      std::string mattype = py::reinterpret_borrow<py::str>(Q.attr("format"));
      if (mattype == "csr") {
        return nmarkov::ctmc_mexp<py::object,marlib::CSRMatrixT>(Q, x, t, trans, params);
      } else if (mattype == "csc") {
        return nmarkov::ctmc_mexp<py::object,marlib::CSCMatrixT>(Q, x, t, trans, params);
      } else if (mattype == "coo") {
        return nmarkov::ctmc_mexp<py::object,marlib::COOMatrixT>(Q, x, t, trans, params);
      } else {
        throw std::runtime_error("Q is either csr, csc and coo.");
      }
    },
    py::arg("Q"), py::arg("x"), py::arg("t"), py::arg("trans"), py::arg("params"),
    "Compute the matrix exponential with uniformization");

  m.def("ctmc_mexpint_dense",
    nmarkov::ctmc_mexpint<dpyarray,marlib::DenseMatrixT>,
    py::arg("Q"), py::arg("x"), py::arg("cx"), py::arg("t"), py::arg("trans"), py::arg("params"),
    "Compute the cumulative value of matrix exponential with uniformization");

  m.def("ctmc_mexpint_sparse",
    [](const py::object& Q, const dpyarray& x, const dpyarray& cx, double t, bool trans, marlib::marlib_params& params) {
      std::string mattype = py::reinterpret_borrow<py::str>(Q.attr("format"));
      if (mattype == "csr") {
        return nmarkov::ctmc_mexpint<py::object,marlib::CSRMatrixT>(Q, x, cx, t, trans, params);
      } else if (mattype == "csc") {
        return nmarkov::ctmc_mexpint<py::object,marlib::CSCMatrixT>(Q, x, cx, t, trans, params);
      } else if (mattype == "coo") {
        return nmarkov::ctmc_mexpint<py::object,marlib::COOMatrixT>(Q, x, cx, t, trans, params);
      } else {
        throw std::runtime_error("Q is either csr, csc and coo.");
      }
    },
    py::arg("Q"), py::arg("x"), py::arg("cx"), py::arg("t"), py::arg("trans"), py::arg("params"),
    "Compute the matrix exponential with uniformization");

  m.def("ctmc_tran_dense",
    nmarkov::ctmc_tran<dpyarray,marlib::DenseMatrixT>,
    py::arg("Q"), py::arg("x"), py::arg("cx"), py::arg("t"), py::arg("trans"), py::arg("params"),
    "Compute the transient solution with uniformization");

  m.def("ctmc_tran_sparse",
    [](const py::object& Q, const dpyarray& x, const dpyarray& cx, const dpyarray& t, bool trans, marlib::marlib_params& params) {
      std::string mattype = py::reinterpret_borrow<py::str>(Q.attr("format"));
      if (mattype == "csr") {
        return nmarkov::ctmc_tran<py::object,marlib::CSRMatrixT>(Q, x, cx, t, trans, params);
      } else if (mattype == "csc") {
        return nmarkov::ctmc_tran<py::object,marlib::CSCMatrixT>(Q, x, cx, t, trans, params);
      } else if (mattype == "coo") {
        return nmarkov::ctmc_tran<py::object,marlib::COOMatrixT>(Q, x, cx, t, trans, params);
      } else {
        throw std::runtime_error("Q is either csr, csc and coo.");
      }
    },
    py::arg("Q"), py::arg("x"), py::arg("cx"), py::arg("t"), py::arg("trans"), py::arg("params"),
    "Compute the transient solution with uniformization");

  m.def("ctmc_tran_rwd_dense",
    nmarkov::ctmc_tran_rwd<dpyarray,marlib::DenseMatrixT>,
    py::arg("Q"), py::arg("x"), py::arg("cx"), py::arg("rwd"), py::arg("t"), py::arg("trans"), py::arg("params"),
    "Compute the transient rewards with uniformization");

  m.def("ctmc_tran_rwd_sparse",
    [](const py::object& Q, const dpyarray& x, const dpyarray& cx, const dpyarray& rwd, const dpyarray& t, bool trans, marlib::marlib_params& params) {
      std::string mattype = py::reinterpret_borrow<py::str>(Q.attr("format"));
      if (mattype == "csr") {
        return nmarkov::ctmc_tran_rwd<py::object,marlib::CSRMatrixT>(Q, x, cx, rwd, t, trans, params);
      } else if (mattype == "csc") {
        return nmarkov::ctmc_tran_rwd<py::object,marlib::CSCMatrixT>(Q, x, cx, rwd, t, trans, params);
      } else if (mattype == "coo") {
        return nmarkov::ctmc_tran_rwd<py::object,marlib::COOMatrixT>(Q, x, cx, rwd, t, trans, params);
      } else {
        throw std::runtime_error("Q is either csr, csc and coo.");
      }
    },
    py::arg("Q"), py::arg("x"), py::arg("cx"), py::arg("rwd"), py::arg("t"), py::arg("trans"), py::arg("params"),
    "Compute the transient rewards with uniformization");

  // m.def("test",
  //   [](const py::object& X) {
  //     using traits1 = marlib::csc_matrix_traits<py::object>;
  //     const int base = traits1::base(X);
  //     const int m = traits1::nrow(X);
  //     const int n = traits1::ncol(X);
  //     const double* valueX = traits1::value(X);
  //     const int* colptr = traits1::colptr(X);
  //     const int* rowind = traits1::rowind(X);

  //     for (int j=0; j<n; j++) {
  //       for (int z=colptr[j]-base; z<colptr[j+1]-base; z++) {
  //         int i = rowind[z] - base;
  //         py::print(i, j, valueX[z]);
  //       }
  //     }
  //   }, "test");
}
