{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };

        shell-pkgs = [
          (pkgs.rust-bin.stable.latest.default.override {
            extensions = [ "rust-src" ];
          })
          # for openblas
          pkgs.openblas

          # for blis (not checked yet)
          # pkgs.blis

          # for netlib
          # pkgs.gfortran
          # pkgs.gfortran.cc
          # pkgs.blas-reference
          # pkgs.lapack-reference

          # for R
          # pkgs.R
          # pkgs.libintl

          pkgs.pkg-config
          pkgs.uv
          pkgs.python313
        ];
        zshCompEnv = pkgs.buildEnv {
          name = "zsh-comp";
          paths = shell-pkgs;
          pathsToLink = [ "/share/zsh" ];
        };
      in {
        devShells.default = pkgs.mkShell rec {
          packages = shell-pkgs;
          ZSH_COMP_FPATH = "${zshCompEnv}/share/zsh/site-functions";
        };
      });
}
