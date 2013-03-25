#include "rate.h"
#include "ofdm.h"

Rate::Rate(int encoding, const Constellation &constellation, const PuncturingMask &puncturingMask, const OFDMFormat &format) :
    encoding(encoding), constellation(constellation), puncturingMask(puncturingMask)
{
    Nbpsc = constellation.Nbpsc;
    Nbps = format.Nsc * Nbpsc * puncturingMask.numerator / puncturingMask.denominator;
    Ncbps = format.Nsc * Nbpsc;
}
