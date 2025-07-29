#!/usr/bin/env bash
#######################################################################################
# Finds the location of this file regardless of where it is called from
#######################################################################################
FILE_LOCATION="${BASH_SOURCE[0]}"
while [ -h "$FILE_LOCATION" ]; do # resolve $FILE_LOCATION until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$FILE_LOCATION" )" >/dev/null 2>&1 && pwd )"
  FILE_LOCATION="$(readlink "$FILE_LOCATION")"
  [[ $FILE_LOCATION != /* ]] && FILE_LOCATION="$DIR/$FILE_LOCATION" # if $FILE_LOCATION was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done

SOURCE_DIR="$( cd -P "$( dirname "$FILE_LOCATION" )" >/dev/null 2>&1 && pwd )"
#######################################################################################

BUILD_TYPE=${BUILD_TYPE:-"Debug"}


OUTPUT_DIR=$PWD
STD=${STD:-"23"}
ARCH=${ARCH:-"x86_64"}
OS=${OS:-"Linux"}

if command -v gcc-14 >/dev/null 2>&1 && command -v g++-14 >/dev/null 2>&1; then
    CC=${CC:-"$(which gcc-14)"}
    CXX=${CXX:-"$(which g++-14)"}
elif command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
    GCC_VERSION=$(gcc -dumpversion | cut -d. -f1)
    if [[ "$GCC_VERSION" -ge 14 ]]; then
        CC=${CC:-"$(which gcc)"}
        CXX=${CXX:-"$(which g++)"}
    else
        echo "Error: gcc/g++ found, but version $GCC_VERSION is less than 14. Please install GCC 14 or newer."
        exit 1
    fi
else
    echo "Error: Neither gcc-14/g++-14 nor a suitable gcc/g++ (version >= 14) found in PATH."
    exit 1
fi


if [[ -f ${OUTPUT_DIR}/conanfile.py ]]; then
    echo "OUTPUT_DIR must not be the same as the source dir"
    exit 1
fi

if [[ ( "${CC}" = "" ) || ( "${CXX}" = "" ) ]]; then
    echo "CC and CXX variables not set. These must be defined to to handle the autodetection".
    exit 1
fi

# export the CC and CXX variables
# they are needed for "conan profile detect"
# and running cmake
export CC 
export CXX

echo $CC
echo $CXX

if [[ $(realpath $(which $CC)) == *"clang"* ]]; then
    COMPILER=clang
else 
    COMPILER=gcc
fi

# Get the compiler version directly from the CC
COMPILER_VERSION=$(${CC} -dumpversion | cut -d. -f1)

PROFILE_NAME=$(basename $(dirname $(realpath ../conanfile.py)))-$(basename $(which ${CC}))

# This generally only needs to be done once
# only calling it becasue its possible the user has never
# called it before
conan profile detect -e

# We are manually setting all the settings because we are dealing with multiple
# compilers and using a very new C++ standard
# normally we'd just call "conan profile detect" to create a profile
conan install ${SOURCE_DIR}/conanfile.py \
    -of ${OUTPUT_DIR} \
    -s compiler=${COMPILER} \
    -s compiler.cppstd=${STD} \
    -s compiler.libcxx=libstdc++11 \
    -s compiler.version=${COMPILER_VERSION} \
    -s:h "&:build_type=${BUILD_TYPE}" \
    -s:h "&:compiler.cppstd=${STD}" \
    --build=missing ${EXTRA_CONAN_ARGS} \
    -s tools.system.package_manager:mode=install \
    -s tools.system.package_manager:sudo=True 

if [[ "$?" != "0" ]]; then
    echo "Failed to run conan"
    exit 1
fi

cmake ${SOURCE_DIR} -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
