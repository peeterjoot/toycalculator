#! /usr/bin/bash

set -x

if [ -e "/etc/debian_version" ] ; then
	version=`dpkg -l antlr4 | sed 's/-.*//'`
else
	version=`rpm -q antlr4 | sed 's/-.*//'`
fi

echo wget https://www.antlr.org/download/antlr-${version}-complete.jar
