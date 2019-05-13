# Use Ubuntu as a parent image
FROM ubuntu:bionic

# Set the working directory to /app/data
WORKDIR /app
RUN mkdir /app/data
ENV DATA=/app/data

# Install dependencies
RUN apt-get update && apt-get install -y \
	bzip2 \
	cmake \
	g++ \
	git \
	libblas-dev \
	libboost-all-dev \
	libbz2-dev \
	libdlib-dev \
	liblapack-dev \
	zlib1g-dev

# Get lloe source from github
RUN git clone https://github.com/jesand/lloe /app/lloe

# Compile lloe
WORKDIR /app/lloe/release
RUN cmake -DCMAKE_BUILD_TYPE=Release ..
RUN make
ENV BIN=/app/lloe/release/src
