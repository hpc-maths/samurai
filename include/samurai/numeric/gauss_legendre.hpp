#pragma once
#include "../cell.hpp"
#include "../static_algorithm.hpp"

namespace samurai
{
    template <std::size_t polynomial_degree = 0>
    class GaussLegendre
    {
      private:

        static constexpr std::size_t n_required_points(int degree)
        {
            return degree < 0 ? N_IMPLEMENTED_POINTS : static_cast<std::size_t>(ce_ceil((degree + 1.) / 2.));
        }

        static constexpr std::size_t N_IMPLEMENTED_POINTS = 20;
        static constexpr std::size_t n_points             = n_required_points(polynomial_degree);

        std::array<double, n_points> points;
        std::array<double, n_points> weights;

      public:

        GaussLegendre()
        {
            init();
        }

        template <std::size_t func_result_size, std::size_t dim, class TInterval, class Func>
        auto quadrature(const Cell<dim, TInterval>& cell, Func&& f)
        {
            const double half_h = cell.length / 2;
            if constexpr (func_result_size == 1)
            {
                double sum = 0;
                compute_quadrature_sum(cell, sum, f);
                return pow(half_h, dim) * sum;
            }
            else
            {
                xt::xtensor_fixed<double, xt::xshape<func_result_size>> sum;
                sum.fill(0);
                compute_quadrature_sum(cell, sum, f);
                return xt::eval(pow(half_h, dim) * sum);
            }
        }

      private:

        template <std::size_t dim, class TInterval, class FuncResultType, class Func>
        void compute_quadrature_sum(const Cell<dim, TInterval>& cell, FuncResultType& sum, Func&& f)
        {
            static_assert(dim >= 1 && dim <= 3,
                          "The Gauss-Legendre quadrature is not implemented "
                          "for this dimension.");

            auto center         = cell.center();
            const double half_h = cell.length / 2;
            decltype(center) eval_point;

            if constexpr (dim == 1)
            {
                for (std::size_t i = 0; i < n_points; ++i)
                {
                    eval_point[0] = half_h * points[i] + center[0];
                    sum += f(eval_point) * weights[i];
                }
            }
            else if constexpr (dim == 2)
            {
                for (std::size_t i = 0; i < n_points; ++i)
                {
                    eval_point[0] = half_h * points[i] + center[0];
                    for (std::size_t j = 0; j < n_points; ++j)
                    {
                        eval_point[1] = half_h * points[j] + center[1];
                        sum += f(eval_point) * weights[i] * weights[j];
                    }
                }
            }
            else if constexpr (dim == 3)
            {
                for (std::size_t i = 0; i < n_points; ++i)
                {
                    eval_point[0] = half_h * points[i] + center[0];
                    for (std::size_t j = 0; j < n_points; ++j)
                    {
                        eval_point[1] = half_h * points[j] + center[1];
                        for (std::size_t k = 0; k < n_points; ++k)
                        {
                            eval_point[2] = half_h * points[k] + center[2];
                            sum += f(eval_point) * weights[i] * weights[j] * weights[k];
                        }
                    }
                }
            }
        }

        void init()
        {
            if constexpr (n_points == 1)
            {
                points[0] = 0;

                weights[0] = 2.0000000000000000;
            }
            else if constexpr (n_points == 2)
            {
                points[0] = -0.5773502691896257;
                points[1] = 0.5773502691896257;

                weights[0] = 1.0000000000000000;
                weights[1] = 1.0000000000000000;
            }
            else if constexpr (n_points == 3)
            {
                points[0] = 0.0000000000000000;
                points[1] = -0.7745966692414834;
                points[2] = 0.7745966692414834;

                weights[0] = 0.8888888888888888;
                weights[1] = 0.5555555555555556;
                weights[2] = 0.5555555555555556;
            }
            else if constexpr (n_points == 4)
            {
                points[0] = -0.3399810435848563;
                points[1] = 0.3399810435848563;
                points[2] = -0.8611363115940526;
                points[3] = 0.8611363115940526;

                weights[0] = 0.6521451548625461;
                weights[1] = 0.6521451548625461;
                weights[2] = 0.3478548451374538;
                weights[3] = 0.3478548451374538;
            }
            else if constexpr (n_points == 5)
            {
                points[0] = 0.0000000000000000;
                points[1] = -0.5384693101056831;
                points[2] = 0.5384693101056831;
                points[3] = -0.9061798459386640;
                points[4] = 0.9061798459386640;

                weights[0] = 0.5688888888888889;
                weights[1] = 0.4786286704993665;
                weights[2] = 0.4786286704993665;
                weights[3] = 0.2369268850561891;
                weights[4] = 0.2369268850561891;
            }
            else if constexpr (n_points == 6)
            {
                points[0] = 0.6612093864662645;
                points[1] = -0.6612093864662645;
                points[2] = -0.2386191860831969;
                points[3] = 0.2386191860831969;
                points[4] = -0.9324695142031521;
                points[5] = 0.9324695142031521;

                weights[0] = 0.3607615730481386;
                weights[1] = 0.3607615730481386;
                weights[2] = 0.4679139345726910;
                weights[3] = 0.4679139345726910;
                weights[4] = 0.1713244923791704;
                weights[5] = 0.1713244923791704;
            }
            else if constexpr (n_points == 7)
            {
                points[0] = 0.0000000000000000;
                points[1] = 0.4058451513773972;
                points[2] = -0.4058451513773972;
                points[3] = -0.7415311855993945;
                points[4] = 0.7415311855993945;
                points[5] = -0.9491079123427585;
                points[6] = 0.9491079123427585;

                weights[0] = 0.4179591836734694;
                weights[1] = 0.3818300505051189;
                weights[2] = 0.3818300505051189;
                weights[3] = 0.2797053914892766;
                weights[4] = 0.2797053914892766;
                weights[5] = 0.1294849661688697;
                weights[6] = 0.1294849661688697;
            }
            else if constexpr (n_points == 8)
            {
                points[0] = -0.1834346424956498;
                points[1] = 0.1834346424956498;
                points[2] = -0.5255324099163290;
                points[3] = 0.5255324099163290;
                points[4] = -0.7966664774136267;
                points[5] = 0.7966664774136267;
                points[6] = -0.9602898564975363;
                points[7] = 0.9602898564975363;

                weights[0] = 0.3626837833783620;
                weights[1] = 0.3626837833783620;
                weights[2] = 0.3137066458778873;
                weights[3] = 0.3137066458778873;
                weights[4] = 0.2223810344533745;
                weights[5] = 0.2223810344533745;
                weights[6] = 0.1012285362903763;
                weights[7] = 0.1012285362903763;
            }
            else if constexpr (n_points == 9)
            {
                points[0] = 0.0000000000000000;
                points[1] = -0.8360311073266358;
                points[2] = 0.8360311073266358;
                points[3] = -0.9681602395076261;
                points[4] = 0.9681602395076261;
                points[5] = -0.3242534234038089;
                points[6] = 0.3242534234038089;
                points[7] = -0.6133714327005904;
                points[8] = 0.6133714327005904;

                weights[0] = 0.3302393550012598;
                weights[1] = 0.1806481606948574;
                weights[2] = 0.1806481606948574;
                weights[3] = 0.0812743883615744;
                weights[4] = 0.0812743883615744;
                weights[5] = 0.3123470770400029;
                weights[6] = 0.3123470770400029;
                weights[7] = 0.2606106964029354;
                weights[8] = 0.2606106964029354;
            }
            else if constexpr (n_points == 10)
            {
                points[0] = -0.1488743389816312;
                points[1] = 0.1488743389816312;
                points[2] = -0.4333953941292472;
                points[3] = 0.4333953941292472;
                points[4] = -0.6794095682990244;
                points[5] = 0.6794095682990244;
                points[6] = -0.8650633666889845;
                points[7] = 0.8650633666889845;
                points[8] = -0.9739065285171717;
                points[9] = 0.9739065285171717;

                weights[0] = 0.2955242247147529;
                weights[1] = 0.2955242247147529;
                weights[2] = 0.2692667193099963;
                weights[3] = 0.2692667193099963;
                weights[4] = 0.2190863625159820;
                weights[5] = 0.2190863625159820;
                weights[6] = 0.1494513491505806;
                weights[7] = 0.1494513491505806;
                weights[8] = 0.0666713443086881;
                weights[9] = 0.0666713443086881;
            }
            else if constexpr (n_points == 11)
            {
                points[0]  = 0.0000000000000000;
                points[1]  = -0.2695431559523450;
                points[2]  = 0.2695431559523450;
                points[3]  = -0.5190961292068118;
                points[4]  = 0.5190961292068118;
                points[5]  = -0.7301520055740494;
                points[6]  = 0.7301520055740494;
                points[7]  = -0.8870625997680953;
                points[8]  = 0.8870625997680953;
                points[9]  = -0.9782286581460570;
                points[10] = 0.9782286581460570;

                weights[0]  = 0.2729250867779006;
                weights[1]  = 0.2628045445102467;
                weights[2]  = 0.2628045445102467;
                weights[3]  = 0.2331937645919905;
                weights[4]  = 0.2331937645919905;
                weights[5]  = 0.1862902109277343;
                weights[6]  = 0.1862902109277343;
                weights[7]  = 0.1255803694649046;
                weights[8]  = 0.1255803694649046;
                weights[9]  = 0.0556685671161737;
                weights[10] = 0.0556685671161737;
            }
            else if constexpr (n_points == 12)
            {
                points[0]  = -0.1252334085114689;
                points[1]  = 0.1252334085114689;
                points[2]  = -0.3678314989981802;
                points[3]  = 0.3678314989981802;
                points[4]  = -0.5873179542866175;
                points[5]  = 0.5873179542866175;
                points[6]  = -0.7699026741943047;
                points[7]  = 0.7699026741943047;
                points[8]  = -0.9041172563704749;
                points[9]  = 0.9041172563704749;
                points[10] = -0.9815606342467192;
                points[11] = 0.9815606342467192;

                weights[0]  = 0.2491470458134028;
                weights[1]  = 0.2491470458134028;
                weights[2]  = 0.2334925365383548;
                weights[3]  = 0.2334925365383548;
                weights[4]  = 0.2031674267230659;
                weights[5]  = 0.2031674267230659;
                weights[6]  = 0.1600783285433462;
                weights[7]  = 0.1600783285433462;
                weights[8]  = 0.1069393259953184;
                weights[9]  = 0.1069393259953184;
                weights[10] = 0.0471753363865118;
                weights[11] = 0.0471753363865118;
            }
            else if constexpr (n_points == 13)
            {
                points[0]  = 0.0000000000000000;
                points[1]  = -0.2304583159551348;
                points[2]  = 0.2304583159551348;
                points[3]  = -0.4484927510364469;
                points[4]  = 0.4484927510364469;
                points[5]  = -0.6423493394403402;
                points[6]  = 0.6423493394403402;
                points[7]  = -0.8015780907333099;
                points[8]  = 0.8015780907333099;
                points[9]  = -0.9175983992229779;
                points[10] = 0.9175983992229779;
                points[11] = -0.9841830547185881;
                points[12] = 0.9841830547185881;

                weights[0]  = 0.2325515532308739;
                weights[1]  = 0.2262831802628972;
                weights[2]  = 0.2262831802628972;
                weights[3]  = 0.2078160475368885;
                weights[4]  = 0.2078160475368885;
                weights[5]  = 0.1781459807619457;
                weights[6]  = 0.1781459807619457;
                weights[7]  = 0.1388735102197872;
                weights[8]  = 0.1388735102197872;
                weights[9]  = 0.0921214998377285;
                weights[10] = 0.0921214998377285;
                weights[11] = 0.0404840047653159;
                weights[12] = 0.0404840047653159;
            }
            else if constexpr (n_points == 14)
            {
                points[0]  = -0.1080549487073437;
                points[1]  = 0.1080549487073437;
                points[2]  = -0.3191123689278897;
                points[3]  = 0.3191123689278897;
                points[4]  = -0.5152486363581541;
                points[5]  = 0.5152486363581541;
                points[6]  = -0.6872929048116855;
                points[7]  = 0.6872929048116855;
                points[8]  = -0.8272013150697650;
                points[9]  = 0.8272013150697650;
                points[10] = -0.9284348836635735;
                points[11] = 0.9284348836635735;
                points[12] = -0.9862838086968123;
                points[13] = 0.9862838086968123;

                weights[0]  = 0.2152638534631578;
                weights[1]  = 0.2152638534631578;
                weights[2]  = 0.2051984637212956;
                weights[3]  = 0.2051984637212956;
                weights[4]  = 0.1855383974779378;
                weights[5]  = 0.1855383974779378;
                weights[6]  = 0.1572031671581935;
                weights[7]  = 0.1572031671581935;
                weights[8]  = 0.1215185706879032;
                weights[9]  = 0.1215185706879032;
                weights[10] = 0.0801580871597602;
                weights[11] = 0.0801580871597602;
                weights[12] = 0.0351194603317519;
                weights[13] = 0.0351194603317519;
            }
            else if constexpr (n_points == 15)
            {
                points[0]  = 0.0000000000000000;
                points[1]  = -0.2011940939974345;
                points[2]  = 0.2011940939974345;
                points[3]  = -0.3941513470775634;
                points[4]  = 0.3941513470775634;
                points[5]  = -0.5709721726085388;
                points[6]  = 0.5709721726085388;
                points[7]  = -0.7244177313601701;
                points[8]  = 0.7244177313601701;
                points[9]  = -0.8482065834104272;
                points[10] = 0.8482065834104272;
                points[11] = -0.9372733924007060;
                points[12] = 0.9372733924007060;
                points[13] = -0.9879925180204854;
                points[14] = 0.9879925180204854;

                weights[0]  = 0.2025782419255613;
                weights[1]  = 0.1984314853271116;
                weights[2]  = 0.1984314853271116;
                weights[3]  = 0.1861610000155622;
                weights[4]  = 0.1861610000155622;
                weights[5]  = 0.1662692058169939;
                weights[6]  = 0.1662692058169939;
                weights[7]  = 0.1395706779261543;
                weights[8]  = 0.1395706779261543;
                weights[9]  = 0.1071592204671719;
                weights[10] = 0.1071592204671719;
                weights[11] = 0.0703660474881081;
                weights[12] = 0.0703660474881081;
                weights[13] = 0.0307532419961173;
                weights[14] = 0.0307532419961173;
            }
            else if constexpr (n_points == 16)
            {
                points[0]  = -0.0950125098376374;
                points[1]  = 0.0950125098376374;
                points[2]  = -0.2816035507792589;
                points[3]  = 0.2816035507792589;
                points[4]  = -0.4580167776572274;
                points[5]  = 0.4580167776572274;
                points[6]  = -0.6178762444026438;
                points[7]  = 0.6178762444026438;
                points[8]  = -0.7554044083550030;
                points[9]  = 0.7554044083550030;
                points[10] = -0.8656312023878318;
                points[11] = 0.8656312023878318;
                points[12] = -0.9445750230732326;
                points[13] = 0.9445750230732326;
                points[14] = -0.9894009349916499;
                points[15] = 0.9894009349916499;

                weights[0]  = 0.1894506104550685;
                weights[1]  = 0.1894506104550685;
                weights[2]  = 0.1826034150449236;
                weights[3]  = 0.1826034150449236;
                weights[4]  = 0.1691565193950025;
                weights[5]  = 0.1691565193950025;
                weights[6]  = 0.1495959888165767;
                weights[7]  = 0.1495959888165767;
                weights[8]  = 0.1246289712555339;
                weights[9]  = 0.1246289712555339;
                weights[10] = 0.0951585116824928;
                weights[11] = 0.0951585116824928;
                weights[12] = 0.0622535239386479;
                weights[13] = 0.0622535239386479;
                weights[14] = 0.0271524594117541;
                weights[15] = 0.0271524594117541;
            }
            else if constexpr (n_points == 17)
            {
                points[0]  = 0.0000000000000000;
                points[1]  = -0.1784841814958479;
                points[2]  = 0.1784841814958479;
                points[3]  = -0.3512317634538763;
                points[4]  = 0.3512317634538763;
                points[5]  = -0.5126905370864769;
                points[6]  = 0.5126905370864769;
                points[7]  = -0.6576711592166907;
                points[8]  = 0.6576711592166907;
                points[9]  = -0.7815140038968014;
                points[10] = 0.7815140038968014;
                points[11] = -0.8802391537269859;
                points[12] = 0.8802391537269859;
                points[13] = -0.9506755217687678;
                points[14] = 0.9506755217687678;
                points[15] = -0.9905754753144174;
                points[16] = 0.9905754753144174;

                weights[0]  = 0.1794464703562065;
                weights[1]  = 0.1765627053669926;
                weights[2]  = 0.1765627053669926;
                weights[3]  = 0.1680041021564500;
                weights[4]  = 0.1680041021564500;
                weights[5]  = 0.1540457610768103;
                weights[6]  = 0.1540457610768103;
                weights[7]  = 0.1351363684685255;
                weights[8]  = 0.1351363684685255;
                weights[9]  = 0.1118838471934040;
                weights[10] = 0.1118838471934040;
                weights[11] = 0.0850361483171792;
                weights[12] = 0.0850361483171792;
                weights[13] = 0.0554595293739872;
                weights[14] = 0.0554595293739872;
                weights[15] = 0.0241483028685479;
                weights[16] = 0.0241483028685479;
            }
            else if constexpr (n_points == 18)
            {
                points[0]  = -0.0847750130417353;
                points[1]  = 0.0847750130417353;
                points[2]  = -0.2518862256915055;
                points[3]  = 0.2518862256915055;
                points[4]  = -0.4117511614628426;
                points[5]  = 0.4117511614628426;
                points[6]  = -0.5597708310739475;
                points[7]  = 0.5597708310739475;
                points[8]  = -0.6916870430603532;
                points[9]  = 0.6916870430603532;
                points[10] = -0.8037049589725231;
                points[11] = 0.8037049589725231;
                points[12] = -0.8926024664975557;
                points[13] = 0.8926024664975557;
                points[14] = -0.9558239495713977;
                points[15] = 0.9558239495713977;
                points[16] = -0.9915651684209309;
                points[17] = 0.9915651684209309;

                weights[0]  = 0.1691423829631436;
                weights[1]  = 0.1691423829631436;
                weights[2]  = 0.1642764837458327;
                weights[3]  = 0.1642764837458327;
                weights[4]  = 0.1546846751262652;
                weights[5]  = 0.1546846751262652;
                weights[6]  = 0.1406429146706507;
                weights[7]  = 0.1406429146706507;
                weights[8]  = 0.1225552067114785;
                weights[9]  = 0.1225552067114785;
                weights[10] = 0.1009420441062872;
                weights[11] = 0.1009420441062872;
                weights[12] = 0.0764257302548891;
                weights[13] = 0.0764257302548891;
                weights[14] = 0.0497145488949698;
                weights[15] = 0.0497145488949698;
                weights[16] = 0.0216160135264833;
                weights[17] = 0.0216160135264833;
            }
            else if constexpr (n_points == 19)
            {
                points[0]  = 0.0000000000000000;
                points[1]  = -0.1603586456402254;
                points[2]  = 0.1603586456402254;
                points[3]  = -0.3165640999636298;
                points[4]  = 0.3165640999636298;
                points[5]  = -0.4645707413759609;
                points[6]  = 0.4645707413759609;
                points[7]  = -0.6005453046616810;
                points[8]  = 0.6005453046616810;
                points[9]  = -0.7209661773352294;
                points[10] = 0.7209661773352294;
                points[11] = -0.8227146565371428;
                points[12] = 0.8227146565371428;
                points[13] = -0.9031559036148179;
                points[14] = 0.9031559036148179;
                points[15] = -0.9602081521348300;
                points[16] = 0.9602081521348300;
                points[17] = -0.9924068438435844;
                points[18] = 0.9924068438435844;

                weights[0]  = 0.1610544498487837;
                weights[1]  = 0.1589688433939543;
                weights[2]  = 0.1589688433939543;
                weights[3]  = 0.1527660420658597;
                weights[4]  = 0.1527660420658597;
                weights[5]  = 0.1426067021736066;
                weights[6]  = 0.1426067021736066;
                weights[7]  = 0.1287539625393362;
                weights[8]  = 0.1287539625393362;
                weights[9]  = 0.1115666455473340;
                weights[10] = 0.1115666455473340;
                weights[11] = 0.0914900216224500;
                weights[12] = 0.0914900216224500;
                weights[13] = 0.0690445427376412;
                weights[14] = 0.0690445427376412;
                weights[15] = 0.0448142267656996;
                weights[16] = 0.0448142267656996;
                weights[17] = 0.0194617882297265;
                weights[18] = 0.0194617882297265;
            }
            else if constexpr (n_points == 20)
            {
                points[0]  = -0.0765265211334973;
                points[1]  = 0.0765265211334973;
                points[2]  = -0.2277858511416451;
                points[3]  = 0.2277858511416451;
                points[4]  = -0.3737060887154195;
                points[5]  = 0.3737060887154195;
                points[6]  = -0.5108670019508271;
                points[7]  = 0.5108670019508271;
                points[8]  = -0.6360536807265150;
                points[9]  = 0.6360536807265150;
                points[10] = -0.7463319064601508;
                points[11] = 0.7463319064601508;
                points[12] = -0.8391169718222188;
                points[13] = 0.8391169718222188;
                points[14] = -0.9122344282513259;
                points[15] = 0.9122344282513259;
                points[16] = -0.9639719272779138;
                points[17] = 0.9639719272779138;
                points[18] = -0.9931285991850949;
                points[19] = 0.9931285991850949;

                weights[0]  = 0.1527533871307258;
                weights[1]  = 0.1527533871307258;
                weights[2]  = 0.1491729864726037;
                weights[3]  = 0.1491729864726037;
                weights[4]  = 0.1420961093183820;
                weights[5]  = 0.1420961093183820;
                weights[6]  = 0.1316886384491766;
                weights[7]  = 0.1316886384491766;
                weights[8]  = 0.1181945319615184;
                weights[9]  = 0.1181945319615184;
                weights[10] = 0.1019301198172404;
                weights[11] = 0.1019301198172404;
                weights[12] = 0.0832767415767048;
                weights[13] = 0.0832767415767048;
                weights[14] = 0.0626720483341091;
                weights[15] = 0.0626720483341091;
                weights[16] = 0.0406014298003869;
                weights[17] = 0.0406014298003869;
                weights[18] = 0.0176140071391521;
                weights[19] = 0.0176140071391521;
            }
        }
    };
}
