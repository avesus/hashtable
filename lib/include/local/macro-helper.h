#ifndef _MACRO_HELPER_H_
#define _MACRO_HELPER_H_

//////////////////////////////////////////////////////////////////////
/* 
   This file contains helper macros for using __VA_ARGS__. Currently
   these macros are written so that 10 arguments is the maximum
   allowed. The formats are clear so expanding them to accept more
   argument should be simply enough.
*/


//////////////////////////////////////////////////////////////////////


//clang-format off
#define PP_NARG(...) \
         PP_NARG_(__VA_ARGS__,PP_RSEQ_N())
#define PP_NARG_(...) \
         PP_ARG_N(__VA_ARGS__)
#define PP_ARG_N( \
          _1, _2, _3, _4, _5, _6, _7, _8, _9,_10, \
         _11,_12,_13,_14,_15,_16,_17,_18,_19,_20, \
         _21,_22,_23,_24,_25,_26,_27,_28,_29,_30, \
         _31,_32,_33,_34,_35,_36,_37,_38,_39,_40, \
         _41,_42,_43,_44,_45,_46,_47,_48,_49,_50, \
         _51,_52,_53,_54,_55,_56,_57,_58,_59,_60, \
         _61,_62,_63,N,...) N
#define PP_RSEQ_N() \
         63,62,61,60,                   \
         59,58,57,56,55,54,53,52,51,50, \
         49,48,47,46,45,44,43,42,41,40, \
         39,38,37,36,35,34,33,32,31,30, \
         29,28,27,26,25,24,23,22,21,20, \
         19,18,17,16,15,14,13,12,11,10, \
         9,8,7,6,5,4,3,2,1,0


#define NARGS_SEQ_2(_1,_1B,_2,_2B,_3,_3B,_4,_4B,_5,_5B,		\
		    _6,_6B,_7,_7B,_8,_8B,_9,_9B,_10,_10B,	\
		    N,...) N
#define NARGS_2(...) NARGS_SEQ_2(__VA_ARGS__,		\
				 10,10,9,9,8,8,7,7,6,6,	\
				 5,5,4,4,3,3,2,2,1,1)

//////////////////////////////////////////////////////////////////////
//Helper for all apply/sum macro
#define PRIMITIVE_CAT(x, y) x ## y
#define CAT(x, y) PRIMITIVE_CAT(x, y)

//////////////////////////////////////////////////////////////////////
//Do macro on all args in __VA_ARGS__
#define APPLY(macro, ...) CAT(APPLY_, PP_NARG(__VA_ARGS__))(macro, __VA_ARGS__)
#define APPLY_1(m, x1) m(x1)
#define APPLY_2(m, x1, x2) m(x1), m(x2)
#define APPLY_3(m, x1, x2, x3) m(x1), m(x2), m(x3)
#define APPLY_4(m, x1, x2, x3, x4) m(x1), m(x2), m(x3), m(x4)
#define APPLY_5(m, x1, x2, x3, x4, x5) m(x1), m(x2), m(x3), m(x4), m(x5)
#define APPLY_6(m, x1, x2, x3, x4, x5, x6) m(x1), m(x2), m(x3), m(x4), m(x5), m(x6)
#define APPLY_7(m, x1, x2, x3, x4, x5, x6, x7) m(x1), m(x2), m(x3), m(x4), m(x5), m(x6), m(x7)
#define APPLY_8(m, x1, x2, x3, x4, x5, x6, x7, x8) m(x1), m(x2), m(x3), m(x4), m(x5), m(x6), m(x7), m(x8)
#define APPLY_9(m, x1, x2, x3, x4, x5, x6, x7, x8, x9) m(x1), m(x2), m(x3), m(x4), m(x5), m(x6), m(x7), m(x8), m(x9)
#define APPLY_10(m, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) m(x1), m(x2), m(x3), m(x4), m(x5), m(x6), m(x7), m(x8), m(x9), m(x10)
#define APPLY_11(m, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11) m(x1), m(x2), m(x3), m(x4), m(x5), m(x6), m(x7), m(x8), m(x9), m(x10), m(x11)
#define APPLY_12(m, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12) m(x1), m(x2), m(x3), m(x4), m(x5), m(x6), m(x7), m(x8), m(x9), m(x10), m(x11), m(x12)
#define APPLY_13(m, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13) m(x1), m(x2), m(x3), m(x4), m(x5), m(x6), m(x7), m(x8), m(x9), m(x10), m(x11), m(x12), m(x13)
#define APPLY_14(m, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14) m(x1), m(x2), m(x3), m(x4), m(x5), m(x6), m(x7), m(x8), m(x9), m(x10), m(x11), m(x12), m(x13), m(x14)
#define APPLY_15(m, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15) m(x1), m(x2), m(x3), m(x4), m(x5), m(x6), m(x7), m(x8), m(x9), m(x10), m(x11), m(x12), m(x13), m(x14), m(x15)
#define APPLY_16(m, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) m(x1), m(x2), m(x3), m(x4), m(x5), m(x6), m(x7), m(x8), m(x9), m(x10), m(x11), m(x12), m(x13), m(x14), m(x15), m(x16)
#define APPLY_17(m, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17) m(x1), m(x2), m(x3), m(x4), m(x5), m(x6), m(x7), m(x8), m(x9), m(x10), m(x11), m(x12), m(x13), m(x14), m(x15), m(x16), m(x17)
#define APPLY_18(m, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18) m(x1), m(x2), m(x3), m(x4), m(x5), m(x6), m(x7), m(x8), m(x9), m(x10), m(x11), m(x12), m(x13), m(x14), m(x15), m(x16), m(x17), m(x18)
#define APPLY_19(m, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19) m(x1), m(x2), m(x3), m(x4), m(x5), m(x6), m(x7), m(x8), m(x9), m(x10), m(x11), m(x12), m(x13), m(x14), m(x15), m(x16), m(x17), m(x18), m(x19)
#define APPLY_20(m, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20) m(x1), m(x2), m(x3), m(x4), m(x5), m(x6), m(x7), m(x8), m(x9), m(x10), m(x11), m(x12), m(x13), m(x14), m(x15), m(x16), m(x17), m(x18), m(x19), m(x20)




//////////////////////////////////////////////////////////////////////
//Sum result of macro on args in __VA_ARGS__
#define SUM(macro, ...) CAT(SUM_, PP_NARG(__VA_ARGS__))(macro, __VA_ARGS__)
#define SUM_1(m, x1) m(x1)
#define SUM_2(m, x1, x2) m(x1) + m(x2)
#define SUM_3(m, x1, x2, x3) m(x1) + m(x2) + m(x3)
#define SUM_4(m, x1, x2, x3, x4) m(x1) + m(x2) + m(x3) + m(x4)
#define SUM_5(m, x1, x2, x3, x4, x5) m(x1) + m(x2) + m(x3) + m(x4) + m(x5)
#define SUM_6(m, x1, x2, x3, x4, x5, x6) m(x1) + m(x2) + m(x3) + m(x4) + m(x5) + m(x6)
#define SUM_7(m, x1, x2, x3, x4, x5, x6, x7) m(x1) + m(x2) + m(x3) + m(x4) + m(x5) + m(x6) + m(x7)
#define SUM_8(m, x1, x2, x3, x4, x5, x6, x7, x8) m(x1) + m(x2) + m(x3) + m(x4) + m(x5) + m(x6) + m(x7) + m(x8)
#define SUM_9(m, x1, x2, x3, x4, x5, x6, x7, x8, x9) m(x1) + m(x2) + m(x3) + m(x4) + m(x5) + m(x6) + m(x7) + m(x8) + m(x9)
#define SUM_10(m, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) m(x1) + m(x2) + m(x3) + m(x4) + m(x5) + m(x6) + m(x7) + m(x8) + m(x9) + m(x10)





//////////////////////////////////////////////////////////////////////
//Do macro which takes common X as first argument for all args in
//__VA_ARGS__
#define APPLY_X(macro, X, ...) CAT(APPLY_X_, PP_NARG(__VA_ARGS__))(macro, X, __VA_ARGS__)
#define APPLY_X_1(m, X, x1) m(X, x1)
#define APPLY_X_2(m, X, x1, x2) m(X, x1), m(X, x2)
#define APPLY_X_3(m, X, x1, x2, x3) m(X, x1), m(X, x2), m(X, x3)
#define APPLY_X_4(m, X, x1, x2, x3, x4) m(X, x1), m(X, x2), m(X, x3), m(X, x4)
#define APPLY_X_5(m, X, x1, x2, x3, x4, x5) m(X, x1), m(X, x2), m(X, x3), m(X, x4), m(X, x5)
#define APPLY_X_6(m, X, x1, x2, x3, x4, x5, x6) m(X, x1), m(X, x2), m(X, x3), m(X, x4), m(X, x5), m(X, x6)
#define APPLY_X_7(m, X, x1, x2, x3, x4, x5, x6, x7) m(X, x1), m(X, x2), m(X, x3), m(X, x4), m(X, x5), m(X, x6), m(X, x7)
#define APPLY_X_8(m, X, x1, x2, x3, x4, x5, x6, x7, x8) m(X, x1), m(X, x2), m(X, x3), m(X, x4), m(X, x5), m(X, x6), m(X, x7), m(X, x8)
#define APPLY_X_9(m, X, x1, x2, x3, x4, x5, x6, x7, x8, x9) m(X, x1), m(X, x2), m(X, x3), m(X, x4), m(X, x5), m(X, x6), m(X, x7), m(X, x8), m(X, x9)
#define APPLY_X_10(m, X, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) m(X, x1), m(X, x2), m(X, x3), m(X, x4), m(X, x5), m(X, x6), m(X, x7), m(X, x8), m(X, x9), m(X, x10)


//////////////////////////////////////////////////////////////////////
//Do macro on all arg pairs in __VA_ARGS__

//Pairs must be ordered first0, second0, first1, second1, ....
#define APPLY_X_N_Y(macro, ...) CAT(APPLY_X_N_Y_, NARGS_2(__VA_ARGS__))(macro, __VA_ARGS__)
#define APPLY_X_N_Y_1(m, x1, y1) m(x1, y1)
#define APPLY_X_N_Y_1B(m, x1, y1) m(x1, y1)
#define APPLY_X_N_Y_2(m, x1, y1, x2, y2) m(x1, y1), m(x2, y2)
#define APPLY_X_N_Y_2B(m, x1, y1, x2, y2) m(x1, y1), m(x2, y2)
#define APPLY_X_N_Y_3(m, x1, y1, x2, y2, x3, y3) m(x1, y1), m(x2, y2), m(x3, y3)
#define APPLY_X_N_Y_3B(m, x1, y1, x2, y2, x3, y3) m(x1, y1), m(x2, y2), m(x3, y3)
#define APPLY_X_N_Y_4(m, x1, y1, x2, y2, x3, y3, x4, y4) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4)
#define APPLY_X_N_Y_4B(m, x1, y1, x2, y2, x3, y3, x4, y4) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4)
#define APPLY_X_N_Y_5(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5)
#define APPLY_X_N_Y_5B(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5)
#define APPLY_X_N_Y_6(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6)
#define APPLY_X_N_Y_6B(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6)
#define APPLY_X_N_Y_7(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6), m(x7, y7)
#define APPLY_X_N_Y_7B(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6), m(x7, y7)
#define APPLY_X_N_Y_8(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6), m(x7, y7), m(x8, y8)
#define APPLY_X_N_Y_8B(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6), m(x7, y7), m(x8, y8)
#define APPLY_X_N_Y_9(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6), m(x7, y7), m(x8, y8), m(x9, y9)
#define APPLY_X_N_Y_9B(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6), m(x7, y7), m(x8, y8), m(x9, y9)
#define APPLY_X_N_Y_10(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6), m(x7, y7), m(x8, y8), m(x9, y9), m(x10, y10)
#define APPLY_X_N_Y_10B(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6), m(x7, y7), m(x8, y8), m(x9, y9), m(x10, y10)


//////////////////////////////////////////////////////////////////////
//Sum macro on all arg pairs in __VA_ARGS__

//Pairs must be ordered first0, second0, first1, second1, ....
#define SUM_X_N_Y(macro, ...) CAT(SUM_X_N_Y_, NARGS_2(__VA_ARGS__))(macro, __VA_ARGS__)
#define SUM_X_N_Y_1(m, x1, y1) m(x1, y1)
#define SUM_X_N_Y_1B(m, x1, y1) m(x1, y1)
#define SUM_X_N_Y_2(m, x1, y1, x2, y2) m(x1, y1) + m(x2, y2)
#define SUM_X_N_Y_2B(m, x1, y1, x2, y2) m(x1, y1) + m(x2, y2)
#define SUM_X_N_Y_3(m, x1, y1, x2, y2, x3, y3) m(x1, y1) + m(x2, y2) + m(x3, y3)
#define SUM_X_N_Y_3B(m, x1, y1, x2, y2, x3, y3) m(x1, y1) + m(x2, y2) + m(x3, y3)
#define SUM_X_N_Y_4(m, x1, y1, x2, y2, x3, y3, x4, y4) m(x1, y1) + m(x2, y2) + m(x3, y3) + m(x4, y4)
#define SUM_X_N_Y_4B(m, x1, y1, x2, y2, x3, y3, x4, y4) m(x1, y1) + m(x2, y2) + m(x3, y3) + m(x4, y4)
#define SUM_X_N_Y_5(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5) m(x1, y1) + m(x2, y2) + m(x3, y3) + m(x4, y4) + m(x5, y5)
#define SUM_X_N_Y_5B(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5) m(x1, y1) + m(x2, y2) + m(x3, y3) + m(x4, y4) + m(x5, y5)
#define SUM_X_N_Y_6(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6) m(x1, y1) + m(x2, y2) + m(x3, y3) + m(x4, y4) + m(x5, y5) + m(x6, y6)
#define SUM_X_N_Y_6B(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6) m(x1, y1) + m(x2, y2) + m(x3, y3) + m(x4, y4) + m(x5, y5) + m(x6, y6)
#define SUM_X_N_Y_7(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7) m(x1, y1) + m(x2, y2) + m(x3, y3) + m(x4, y4) + m(x5, y5) + m(x6, y6) + m(x7, y7)
#define SUM_X_N_Y_7B(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7) m(x1, y1) + m(x2, y2) + m(x3, y3) + m(x4, y4) + m(x5, y5) + m(x6, y6) + m(x7, y7)
#define SUM_X_N_Y_8(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8) m(x1, y1) + m(x2, y2) + m(x3, y3) + m(x4, y4) + m(x5, y5) + m(x6, y6) + m(x7, y7) + m(x8, y8)
#define SUM_X_N_Y_8B(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8) m(x1, y1) + m(x2, y2) + m(x3, y3) + m(x4, y4) + m(x5, y5) + m(x6, y6) + m(x7, y7) + m(x8, y8)
#define SUM_X_N_Y_9(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9) m(x1, y1) + m(x2, y2) + m(x3, y3) + m(x4, y4) + m(x5, y5) + m(x6, y6) + m(x7, y7) + m(x8, y8) + m(x9, y9)
#define SUM_X_N_Y_9B(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9) m(x1, y1) + m(x2, y2) + m(x3, y3) + m(x4, y4) + m(x5, y5) + m(x6, y6) + m(x7, y7) + m(x8, y8) + m(x9, y9)
#define SUM_X_N_Y_10(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10) m(x1, y1) + m(x2, y2) + m(x3, y3) + m(x4, y4) + m(x5, y5) + m(x6, y6) + m(x7, y7) + m(x8, y8) + m(x9, y9) + m(x10, y10)
#define SUM_X_N_Y_10B(m, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10) m(x1, y1) + m(x2, y2) + m(x3, y3) + m(x4, y4) + m(x5, y5) + m(x6, y6) + m(x7, y7) + m(x8, y8) + m(x9, y9) + m(x10, y10)


//////////////////////////////////////////////////////////////////////
//Do macro on all arg pairs in __VA_ARGS__ with X as common first
//argument.

//Pairs must be ordered first0, second0, first1, second1, ....
#define APPLY_X_Y_Z(macro, X, ...) CAT(APPLY_X_Y_Z_, NARGS_2(__VA_ARGS__))(macro, X, __VA_ARGS__)
#define APPLY_X_Y_Z_1(m, X, x1, y1) m(X, x1, y1)
#define APPLY_X_Y_Z_1B(m, X, x1, y1) m(X, x1, y1)
#define APPLY_X_Y_Z_2(m, X, x1, y1, x2, y2) m(X, x1, y1), m(X, x2, y2)
#define APPLY_X_Y_Z_2B(m, X, x1, y1, x2, y2) m(X, x1, y1), m(X, x2, y2)
#define APPLY_X_Y_Z_3(m, X, x1, y1, x2, y2, x3, y3) m(X, x1, y1), m(X, x2, y2), m(X, x3, y3)
#define APPLY_X_Y_Z_3B(m, X, x1, y1, x2, y2, x3, y3) m(X, x1, y1), m(X, x2, y2), m(X, x3, y3)
#define APPLY_X_Y_Z_4(m, X, x1, y1, x2, y2, x3, y3, x4, y4) m(X, x1, y1), m(X, x2, y2), m(X, x3, y3), m(X, x4, y4)
#define APPLY_X_Y_Z_4B(m, X, x1, y1, x2, y2, x3, y3, x4, y4) m(X, x1, y1), m(X, x2, y2), m(X, x3, y3), m(X, x4, y4)
#define APPLY_X_Y_Z_5(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5) m(X, x1, y1), m(X, x2, y2), m(X, x3, y3), m(X, x4, y4), m(X, x5, y5)
#define APPLY_X_Y_Z_5B(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5) m(X, x1, y1), m(X, x2, y2), m(X, x3, y3), m(X, x4, y4), m(X, x5, y5)
#define APPLY_X_Y_Z_6(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6) m(X, x1, y1), m(X, x2, y2), m(X, x3, y3), m(X, x4, y4), m(X, x5, y5), m(X, x6, y6)
#define APPLY_X_Y_Z_6B(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6) m(X, x1, y1), m(X, x2, y2), m(X, x3, y3), m(X, x4, y4), m(X, x5, y5), m(X, x6, y6)
#define APPLY_X_Y_Z_7(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7) m(X, x1, y1), m(X, x2, y2), m(X, x3, y3), m(X, x4, y4), m(X, x5, y5), m(X, x6, y6), m(X, x7, y7)
#define APPLY_X_Y_Z_7B(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7) m(X, x1, y1), m(X, x2, y2), m(X, x3, y3), m(X, x4, y4), m(X, x5, y5), m(X, x6, y6), m(X, x7, y7)
#define APPLY_X_Y_Z_8(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8) m(X, x1, y1), m(X, x2, y2), m(X, x3, y3), m(X, x4, y4), m(X, x5, y5), m(X, x6, y6), m(X, x7, y7), m(X, x8, y8)
#define APPLY_X_Y_Z_8B(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8) m(X, x1, y1), m(X, x2, y2), m(X, x3, y3), m(X, x4, y4), m(X, x5, y5), m(X, x6, y6), m(X, x7, y7), m(X, x8, y8)
#define APPLY_X_Y_Z_9(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9) m(X, x1, y1), m(X, x2, y2), m(X, x3, y3), m(X, x4, y4), m(X, x5, y5), m(X, x6, y6), m(X, x7, y7), m(X, x8, y8), m(X, x9, y9)
#define APPLY_X_Y_Z_9B(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9) m(X, x1, y1), m(X, x2, y2), m(X, x3, y3), m(X, x4, y4), m(X, x5, y5), m(X, x6, y6), m(X, x7, y7), m(X, x8, y8), m(X, x9, y9)
#define APPLY_X_Y_Z_10(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10) m(X, x1, y1), m(X, x2, y2), m(X, x3, y3), m(X, x4, y4), m(X, x5, y5), m(X, x6, y6), m(X, x7, y7), m(X, x8, y8), m(X, x9, y9), m(X, x10, y10)
#define APPLY_X_Y_Z_10B(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10) m(X, x1, y1), m(X, x2, y2), m(X, x3, y3), m(X, x4, y4), m(X, x5, y5), m(X, x6, y6), m(X, x7, y7), m(X, x8, y8), m(X, x9, y9), m(X, x10, y10)


//////////////////////////////////////////////////////////////////////
//This is current NOT used but might be useful to have and is not
//something I hope to rewrite

//Do macro on all arg pairs in __VA_ARGS__

//Pairs are ordered such args args [0, N/2) are first0-N, and args
//[N/2, N) are second0-N
#define APPLY_X_Y(macro, F, V) CAT(APPLY_X_Y_, PP_NARG(F))(macro, F, V)
#define APPLY_X_Y_1(m, x1, y1) m(x1, y1)
#define APPLY_X_Y_1B(m, x1, y1) m(x1, y1)
#define APPLY_X_Y_2(m, x1, x2, y1, y2) m(x1, y1), m(x2, y2)
#define APPLY_X_Y_2B(m, x1, x2, y1, y2) m(x1, y1), m(x2, y2)
#define APPLY_X_Y_3(m, x1, x2, x3, y1, y2 , y3) m(x1, y1), m(x2, y2), m(x3, y3)
#define APPLY_X_Y_3B(m, x1, x2, x3, y1, y2 , y3) m(x1, y1), m(x2, y2), m(x3, y3)
#define APPLY_X_Y_4(m, x1, x2, x3, x4, y1, y2 , y3, y4) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4)
#define APPLY_X_Y_4B(m, x1, x2, x3, x4, y1, y2 , y3, y4) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4)
#define APPLY_X_Y_5(m, x1, x2, x3, x4, x5, y1, y2 , y3, y4, y5) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5)
#define APPLY_X_Y_5B(m, x1, x2, x3, x4, x5, y1, y2 , y3, y4, y5) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5)
#define APPLY_X_Y_5B(m, x1, x2, x3, x4, x5, y1, y2 , y3, y4, y5) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5)
#define APPLY_X_Y_6(m, x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6)
#define APPLY_X_Y_6B(m, x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6)
#define APPLY_X_Y_7(m, x1, x2, x3, x4, x5, x6, x7, y1, y2, y3, y4, y5, y6, y7) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6), m(x7, y7)
#define APPLY_X_Y_7B(m, x1, x2, x3, x4, x5, x6, x7, y1, y2, y3, y4, y5, y6, y7) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6), m(x7, y7)
#define APPLY_X_Y_8(m, x1, x2, x3, x4, x5, x6, x7, x8, y1, y2, y3, y4, y5, y6, y7, y8) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6), m(x7, y7), m(x8, y8)
#define APPLY_X_Y_8B(m, x1, x2, x3, x4, x5, x6, x7, x8, y1, y2, y3, y4, y5, y6, y7, y8) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6), m(x7, y7), m(x8, y8)
#define APPLY_X_Y_9(m, x1, x2, x3, x4, x5, x6, x7, x8, x9, y1, y2, y3, y4, y5, y6, y7, y8, y9) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6), m(x7, y7), m(x8, y8), m(x9, y9)
#define APPLY_X_Y_9B(m, x1, x2, x3, x4, x5, x6, x7, x8, x9, y1, y2, y3, y4, y5, y6, y7, y8, y9) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6), m(x7, y7), m(x8, y8), m(x9, y9)
#define APPLY_X_Y_10(m, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6), m(x7, y7), m(x8, y8), m(x9, y9), m(x10, y10)
#define APPLY_X_Y_10B(m, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10) m(x1, y1), m(x2, y2), m(x3, y3), m(x4, y4), m(x5, y5), m(x6, y6), m(x7, y7), m(x8, y8), m(x9, y9), m(x10, y10)



//////////////////////////////////////////////////////////////////////
//This is current NOT used but might be useful to have and is not
//something I hope to rewrite

//Do macro on all arg pairs in __VA_ARGS__ with X as common first
//argument.

//Pairs must be ordered first0, second0, first1, second1, ....

//This configuration seperates each macro invocation by ';' instead of
//',' which is necessary for if the macro to apply is multi-line.
#define APPLY_X_Y_Z_E(macro, X, ...) CAT(APPLY_X_Y_Z_E_, NARGS_2(__VA_ARGS__))(macro, X, __VA_ARGS__)
#define APPLY_X_Y_Z_E_1(m, X, x1, y1) m(X, x1, y1)
#define APPLY_X_Y_Z_E_1B(m, X, x1, y1) m(X, x1, y1)
#define APPLY_X_Y_Z_E_2(m, X, x1, y1, x2, y2) m(X, x1, y1); m(X, x2, y2)
#define APPLY_X_Y_Z_E_2B(m, X, x1, y1, x2, y2) m(X, x1, y1); m(X, x2, y2)
#define APPLY_X_Y_Z_E_3(m, X, x1, y1, x2, y2, x3, y3) m(X, x1, y1); m(X, x2, y2); m(X, x3, y3)
#define APPLY_X_Y_Z_E_3B(m, X, x1, y1, x2, y2, x3, y3) m(X, x1, y1); m(X, x2, y2); m(X, x3, y3)
#define APPLY_X_Y_Z_E_4(m, X, x1, y1, x2, y2, x3, y3, x4, y4) m(X, x1, y1); m(X, x2, y2); m(X, x3, y3); m(X, x4, y4)
#define APPLY_X_Y_Z_E_4B(m, X, x1, y1, x2, y2, x3, y3, x4, y4) m(X, x1, y1); m(X, x2, y2); m(X, x3, y3); m(X, x4, y4)
#define APPLY_X_Y_Z_E_5(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5) m(X, x1, y1); m(X, x2, y2); m(X, x3, y3); m(X, x4, y4); m(X, x5, y5)
#define APPLY_X_Y_Z_E_5B(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5) m(X, x1, y1); m(X, x2, y2); m(X, x3, y3); m(X, x4, y4); m(X, x5, y5)
#define APPLY_X_Y_Z_E_6(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6) m(X, x1, y1); m(X, x2, y2); m(X, x3, y3); m(X, x4, y4); m(X, x5, y5); m(X, x6, y6)
#define APPLY_X_Y_Z_E_6B(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6) m(X, x1, y1); m(X, x2, y2); m(X, x3, y3); m(X, x4, y4); m(X, x5, y5); m(X, x6, y6)
#define APPLY_X_Y_Z_E_7(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7) m(X, x1, y1); m(X, x2, y2); m(X, x3, y3); m(X, x4, y4); m(X, x5, y5); m(X, x6, y6); m(X, x7, y7)
#define APPLY_X_Y_Z_E_7B(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7) m(X, x1, y1); m(X, x2, y2); m(X, x3, y3); m(X, x4, y4); m(X, x5, y5); m(X, x6, y6); m(X, x7, y7)
#define APPLY_X_Y_Z_E_8(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8) m(X, x1, y1); m(X, x2, y2); m(X, x3, y3); m(X, x4, y4); m(X, x5, y5); m(X, x6, y6); m(X, x7, y7); m(X, x8, y8)
#define APPLY_X_Y_Z_E_8B(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8) m(X, x1, y1); m(X, x2, y2); m(X, x3, y3); m(X, x4, y4); m(X, x5, y5); m(X, x6, y6); m(X, x7, y7); m(X, x8, y8)
#define APPLY_X_Y_Z_E_9(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9) m(X, x1, y1); m(X, x2, y2); m(X, x3, y3); m(X, x4, y4); m(X, x5, y5); m(X, x6, y6); m(X, x7, y7); m(X, x8, y8); m(X, x9, y9)
#define APPLY_X_Y_Z_E_9B(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9) m(X, x1, y1); m(X, x2, y2); m(X, x3, y3); m(X, x4, y4); m(X, x5, y5); m(X, x6, y6); m(X, x7, y7); m(X, x8, y8); m(X, x9, y9)
#define APPLY_X_Y_Z_E_10(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10) m(X, x1, y1); m(X, x2, y2); m(X, x3, y3); m(X, x4, y4); m(X, x5, y5); m(X, x6, y6); m(X, x7, y7); m(X, x8, y8); m(X, x9, y9); m(X, x10, y10)
#define APPLY_X_Y_Z_E_10B(m, X, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10) m(X, x1, y1); m(X, x2, y2); m(X, x3, y3); m(X, x4, y4); m(X, x5, y5); m(X, x6, y6); m(X, x7, y7); m(X, x8, y8); m(X, x9, y9); m(X, x10, y10)

//clang-format on
#endif

