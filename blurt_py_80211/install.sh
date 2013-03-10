#!/bin/bash
clang -o /usr/bin/blurt wrapper.c
chown peteriannucci /usr/bin/blurt
chmod 6755 /usr/bin/blurt
