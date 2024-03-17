# Elliptic Curves

Author: [Eta](https://twitter.com/pwhattie)

In zk-SNARKs, points on an elliptic curve are obtained by encrypting polynomials with coefficients and values over a finite field. By converting polynomial equations in QAP to relationships between points on an elliptic curve, zero-knowledge proofs can be realized.

This encryption ensures that the prover can only perform linear operations, because based on the KEA (Knowledge of Exponent Assumption) hypothesis, if two points on an elliptic curve satisfy a certain relationship, then there must exist a number such that one point is equal to a multiple of the other point.

The operations on elliptic curves, simply put, include the geometry and coordinate operations of elliptic curve addition. Assuming P = $(x_1, y_1),$ Q = $(x_2, y_2):$ there are two cases, P ≠ Q connecting two points with a straight line and P = Q making a tangent at one point, and each case has two cases, the intersection point is O or the intersection point is R:

1. Geometric operations: The sum of three points is equal to the infinite point O, the sum of two points is equal to the inverse of the intersection point O or -R: P + Q + O = O ⇒ P + Q = O, P + Q + R = O ⇒ P + Q = -R;

2. Coordinate operations: Let the equation of the straight line L of the two points P and Q be y = λx + c, calculate the slope λ and intercept c of L, and thus calculate the coordinates of the inverse of R $(x_3, - y_3)$ = $(λ^2 - x_1 -x_2, -λ^3 + (x_1 + x_2)λ -c);$
