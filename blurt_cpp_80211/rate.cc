#include "rate.h"
#include "ofdm.h"

Rate::Rate(uint32_t encoding_, const Constellation &constellation_, const PuncturingMask &puncturingMask_, const OFDMFormat &format) :
    encoding(encoding_), constellation(constellation_), puncturingMask(puncturingMask_)
{
    Nbpsc = constellation.Nbpsc;
    Nbps = format.Nsc * Nbpsc * puncturingMask.numerator / puncturingMask.denominator;
    Ncbps = format.Nsc * Nbpsc;
}
