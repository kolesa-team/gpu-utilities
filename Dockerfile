ARG cuda_version=11.6.0
ARG ubuntu_version=20.04

FROM nvidia/cuda:${cuda_version}-devel-ubuntu${ubuntu_version}

ARG python_version=3.7.16
ARG requirements_dependencies=development
ARG workdir="/app"

WORKDIR ${workdir}

ENV APT_FLAGS_PERSISTENT="--yes --no-install-recommends" 
ENV PYENV_URL="https://github.com/pyenv/pyenv.git" 		
ENV DEBIAN_FRONTEND="noninteractive" 					
ENV PYTHONPATH="$PYTHONPATH:${workdir}" 				
ENV PYENV_ROOT="${workdir}/.pyenv" 						
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"


# Install pyenv
RUN apt-get update 									\
	&& set -x 										\
	&& apt-get ${APT_FLAGS_PERSISTENT} install ca-certificates git \
    # Pyenv needs these dependencies to install Python versions
    # See: https://github.com/pyenv/pyenv/wiki#suggested-build-environment
    && apt-get ${APT_FLAGS_PERSISTENT} install make \
		build-essential 							\
		libssl-dev 									\
		zlib1g-dev 									\
		libbz2-dev 									\
		libreadline-dev 							\
		libsqlite3-dev 								\
		wget 										\
		curl 										\
		llvm 										\ 
		libncurses5-dev 							\
		xz-utils 									\
		tk-dev 										\
		libxml2-dev 								\
		libxmlsec1-dev 								\
		libffi-dev 									\
		liblzma-dev 								\
    && git clone --depth=1 ${PYENV_URL} .pyenv 		\
    && pyenv install ${python_version} 				\
    && pyenv global ${python_version} 				

COPY requirements/ requirements/

RUN pip install --no-cache-dir -r "requirements/${requirements_dependencies}.txt"

COPY . .

ENTRYPOINT python main.py


