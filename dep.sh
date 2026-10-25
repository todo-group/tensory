#!/bin/sh

cargo tree | grep -v '(\*)' | grep -v -- '-dependencies'  | sed '/^$/d' | sed 's/^.*─ //g' | sort -u
