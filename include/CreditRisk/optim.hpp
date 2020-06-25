#ifndef OPTIM_HPP
#define OPTIM_HPP

#include <iostream>
#include <math.h>
#include <float.h>

namespace CreditRisk {
    namespace Optim {

        template<class type, class T, class T1, class... Args>
        auto root_secant(type T::*f, T1& obj, double x1, double x2, double tol, Args&&... args)
        {
            double k,f1,f2, xm1(0) ,xm2(0);;

            f1 = (static_cast<T &&>(obj).*f)(x1, std::forward<Args>(args)...);
            f2 = (static_cast<T &&>(obj).*f)(x2, std::forward<Args>(args)...);

            if (f1 * f2 >= 0) return (fabs(f1) < fabs(f2)) ? x1 : x2;

            do {
                xm1 = (x1 * f2 - x2 * f1) / (f2 - f1);
                k = (static_cast<T &&>(obj).*f)(x1, std::forward<Args>(args)...) * (static_cast<T &&>(obj).*f)(xm1, std::forward<Args>(args)...);

                x1 = x2;
                x2 = xm1;

                if (fabs(k) < tol) return xm1;

                xm2 = (x1 * f2 - x2 * f1) / (f2 - f1);

                f1 = (static_cast<T &&>(obj).*f)(x1, std::forward<Args>(args)...);
                f2 = (static_cast<T &&>(obj).*f)(x2, std::forward<Args>(args)...);
            } while (fabs(xm2 - xm1) >= 1e-9);

            return xm1;
        }

        template<class type, class T, class T1, class... Args>
        auto root_bisection(type T::*f, T1& obj, double ax, double bx, double tol, bool left, Args&&... args)
        {
            double fa, fb, fc, c(ax);

            fa = (static_cast<T &&>(obj).*f)(ax, std::forward<Args>(args)...);
            fb = (static_cast<T &&>(obj).*f)(bx, std::forward<Args>(args)...);

            if (fa * fb >= 0) return (fabs(fa) < fabs(fb)) ? ax : bx;

            while ((bx - ax) >= 1e-9)
            {
                c = (ax + bx) / 2;
                fc = (static_cast<T &&>(obj).*f)(c, std::forward<Args>(args)...);
                if (fabs(fc) < tol) return c;;

                if (fc * (static_cast<T &&>(obj).*f)(ax, std::forward<Args>(args)...) < 0)
                {
                    bx = c;
                } else {
                    ax = c;
                }
            }

            if (left) return (ax < bx ? ax : bx);

            return (ax < bx ? bx: ax);
        }

        template<class type, class T, class T1, class... Args>
        auto root_Newton(type T::*f, type T::*df, T1&& obj, double x0, double tol, Args&&... args)
        {
                double fx = (static_cast<T &&>(obj).*f)(x0, std::forward<Args>(args)...);

                while (abs(fx) > tol)
                {
                        x0 -= fx / (static_cast<T &&>(obj).*df)(x0, std::forward<Args>(args)...);
                        fx = (static_cast<T &&>(obj).*f)(x0, std::forward<Args>(args)...);
                }

                return x0;
        }

        // Brent R implementation

        template<class type, class T, class T1, class... Args>
        auto root_Brent(type T::*f, T1&& obj, double ax, double bx, double tol, Args&&... args)
        {
            /*  c is the squared inverse of the golden ratio */
            const double c = (3. - sqrt(5.)) * .5;

            /* Local variables */
            double a, b, d, e, p, q, r, u, v, w, x;
            double t2, fu, fv, fw, fx, xm, eps, tol1, tol3;

            /*  eps is approximately the square root of the relative machine precision. */
            eps = DBL_EPSILON;
            tol1 = eps + 1.;/* the smallest 1.000... > 1 */
            eps = sqrt(eps);

            a = ax;
            b = bx;
            v = a + c * (b - a);
            w = v;
            x = v;

            d = 0.;/* -Wall */
            e = 0.;
            fx = (static_cast<T &&>(obj).*f)(x, std::forward<Args>(args)...);
            fv = fx;
            fw = fx;
            tol3 = tol / 3.;

            /*  main loop starts here ----------------------------------- */

            for (;;) {
                xm = (a + b) * .5;
                tol1 = eps * fabs(x) + tol3;
                t2 = tol1 * 2.;

                /* check stopping criterion */

                if (fabs(x - xm) <= t2 - (b - a) * .5) break;
                p = 0.;
                q = 0.;
                r = 0.;
                if (fabs(e) > tol1) { /* fit parabola */

                    r = (x - w) * (fx - fv);
                    q = (x - v) * (fx - fw);
                    p = (x - v) * q - (x - w) * r;
                    q = (q - r) * 2.;
                    if (q > 0.) p = -p; else q = -q;
                    r = e;
                    e = d;
                }

                if (fabs(p) >= fabs(q * .5 * r) ||
                    p <= q * (a - x) || p >= q * (b - x)) { /* a golden-section step */

                    if (x < xm) e = b - x; else e = a - x;
                    d = c * e;
                }
                else { /* a parabolic-interpolation step */

                    d = p / q;
                    u = x + d;

                    /* f must not be evaluated too close to ax or bx */

                    if (u - a < t2 || b - u < t2) {
                        d = tol1;
                        if (x >= xm) d = -d;
                    }
                }

                /* f must not be evaluated too close to x */

                if (fabs(d) >= tol1)
                    u = x + d;
                else if (d > 0.)
                    u = x + tol1;
                else
                    u = x - tol1;

                fu = (static_cast<T &&>(obj).*f)(u, std::forward<Args>(args)...);

                /*  update  a, b, v, w, and x */

                if (fu <= fx) {
                    if (u < x) b = x; else a = x;
                    v = w;    w = x;   x = u;
                    fv = fw; fw = fx; fx = fu;
                }
                else {
                    if (u < x) a = u; else b = u;
                    if (fu <= fw || w == x) {
                        v = w; fv = fw;
                        w = u; fw = fu;
                    }
                    else if (fu <= fv || v == x || v == w) {
                        v = u; fv = fu;
                    }
                }
            }
            /* end of main loop */

            return x;
        }

        // Brentq scipy implementation

        template<class type, class T, class T1, class... Args>
        auto root_Brentq(type T::*f, T1&& obj, double xa, double xb, double xtol, double rtol, int iter, Args&&... args)
        {
            double xpre = xa, xcur = xb;
            double xblk = 0., fpre, fcur, fblk = 0., spre = 0., scur = 0., sbis;
            /* the tolerance is 2*delta */
            double delta;
            double stry, dpre, dblk;
            int i;

            fpre = -(static_cast<T &&>(obj).*f)(xpre, std::forward<Args>(args)...);
            fcur = -(static_cast<T &&>(obj).*f)(xcur, std::forward<Args>(args)...);

            if (fpre * fcur >= 0)
            {
                return (fabs(fpre) < fabs(fcur)) ? xa : xb;
            }

            for (i = 0; i < iter; i++) {
                if (fpre*fcur < 0) {
                    xblk = xpre;
                    fblk = fpre;
                    spre = scur = xcur - xpre;
                }
                if (fabs(fblk) < fabs(fcur)) {
                    xpre = xcur;
                    xcur = xblk;
                    xblk = xpre;

                    fpre = fcur;
                    fcur = fblk;
                    fblk = fpre;
                }

                delta = (xtol + rtol*fabs(xcur)) / 2;
                sbis = (xblk - xcur) / 2;
                if (fcur == 0 || fabs(sbis) < delta) {
                    return xcur;
                }

                if (fabs(spre) > delta && fabs(fcur) < fabs(fpre)) {
                    if (xpre == xblk) {
                        /* interpolate */
                        stry = -fcur*(xcur - xpre) / (fcur - fpre);
                    }
                    else {
                        /* extrapolate */
                        dpre = (fpre - fcur) / (xpre - xcur);
                        dblk = (fblk - fcur) / (xblk - xcur);
                        stry = -fcur*(fblk*dblk - fpre*dpre)
                            / (dblk*dpre*(fblk - fpre));
                    }
                    if (2 * fabs(stry) < std::min(fabs(spre), 3 * fabs(sbis) - delta)) {
                        /* good short step */
                        spre = scur;
                        scur = stry;
                    }
                    else {
                        /* bisect */
                        spre = sbis;
                        scur = sbis;
                    }
                }
                else {
                    /* bisect */
                    spre = sbis;
                    scur = sbis;
                }

                xpre = xcur; fpre = fcur;
                if (fabs(scur) > delta) {
                    xcur += scur;
                }
                else {
                    xcur += (sbis > 0 ? delta : -delta);
                }

                fcur = -(static_cast<T &&>(obj).*f)(xcur, std::forward<Args>(args)...);
            }

            return xcur;
        }
    }
}

#endif // OPTIM_HPP
