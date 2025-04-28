#! /usr/bin/bash

set -x

# Only need the jar download for WSL, where the antlr4 runtime version, doesn't match the antlr4 driver itself:
#
#ii  antlr4                                 4.9.2-2                                 all          ANTLR Parser Generator
#ii  libantlr4-runtime4.10:amd64            4.10+dfsg-1                             amd64        ANTLR Parser Generator - C++ runtime support (shared library)
#
# Everywhere else, just use the installed anltr4 driver.

#if [ "${WSL_DISTRO_NAME}" != "" ] ; then
    version=`dpkg -l | grep antlr4-runtime4 | sed 's/.*:amd64 *//;s/+.*//'`
#else
#if [ -e "/etc/debian_version" ] ; then
#	version=`dpkg -l antlr4 | sed 's/-.*//'`
#else
#	version=`rpm -q antlr4 | sed 's/antlr4-//;s/-.*//;'`
#fi
#fi

wget https://www.antlr.org/download/antlr-${version}-complete.jar
