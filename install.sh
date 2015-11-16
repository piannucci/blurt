#!/bin/bash
if which -s port
then
    echo "Found MacPorts"
else
    echo "Installing MacPorts"
    VERS=`sw_vers -productVersion | cut -d . -f 1-2`
    case $VERS in
        10.9) VERNAME=Mavericks;;
        10.10) VERNAME=Yosemite;;
        10.11) VERNAME=ElCapitan;;
    esac
    PACKAGE=MacPorts-2.3.4-$VERS-$VERNAME.pkg
    URL=http://iweb.dl.sourceforge.net/project/macports/MacPorts/2.3.4/$PACKAGE
    curl $URL -o $PACKAGE
    open -W $PACKAGE
fi

export PATH="/opt/local/bin:/opt/local/sbin:$PATH"

sudo port selfupdate
sudo port install python3.4 py34-numpy py34-cython
mkdir blurt
pushd blurt
git clone https://github.com/piannucci/weave.git
pushd weave
git checkout python3
sudo python3.4 setup.py install
popd
git clone https://github.com/piannucci/audio.git
pushd audio
sudo python3.4 setup.py install
popd
git clone https://github.com/piannucci/iir.git
pushd iir
sudo python3.4 setup.py install
popd
git clone https://github.com/piannucci/blurt.git
pushd blurt
cd blurt_py_80211/streaming
./wifi80211.py
popd
popd
