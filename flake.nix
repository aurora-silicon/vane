{
  description = "vane — Lean 4 compiler and safety verifier for Linux accel/ane";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system nixpkgs.legacyPackages.${system});
    in
    {
      packages = forAllSystems (system: pkgs:
        let
          vane = pkgs.stdenv.mkDerivation {
            pname = "vane";
            version = "0.1.0";
            src = self;

            nativeBuildInputs = [ pkgs.lean4 ];
            dontConfigure = true;

            buildPhase = ''
              runHook preBuild
              export HOME="$TMPDIR"
              lake build
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              install -Dm755 .lake/build/bin/vane "$out/bin/vane"
              install -Dm755 .lake/build/bin/vane-test "$out/bin/vane-test"
              runHook postInstall
            '';

            meta = with pkgs.lib; {
              description = "Lean 4 compiler and safety verifier for Linux accel/ane";
              mainProgram = "vane";
              license = licenses.mit;
              platforms = systems;
            };
          };

          vane-runner = pkgs.buildGoModule {
            pname = "vane-runner";
            version = "0.1.0";
            src = self;
            vendorHash = null;
            subPackages = [ "cmd/vane-runner" ];
            env.CGO_ENABLED = "0";
            doCheck = false;

            meta = with pkgs.lib; {
              description = "Zero-dependency Go device runner and reference harness for Linux accel/ane";
              mainProgram = "vane-runner";
              license = licenses.mit;
              platforms = systems;
            };
          };

          vane-runner-arm64 = pkgs.stdenv.mkDerivation {
            pname = "vane-runner-arm64";
            version = "0.1.0";
            src = self;

            nativeBuildInputs = [ pkgs.go ];
            dontConfigure = true;

            buildPhase = ''
              runHook preBuild
              export HOME="$TMPDIR"
              export GOCACHE="$TMPDIR/go-cache"
              CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -trimpath -o vane-runner ./cmd/vane-runner
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              install -Dm755 vane-runner "$out/bin/vane-runner"
              runHook postInstall
            '';

            meta = with pkgs.lib; {
              description = "Static Linux arm64 device runner for Linux accel/ane";
              mainProgram = "vane-runner";
              license = licenses.mit;
              platforms = systems;
            };
          };
        in
        {
          default = vane;
          inherit vane vane-runner vane-runner-arm64;
        });

      devShells = forAllSystems (system: pkgs: {
        default = pkgs.mkShell {
          packages = [
            pkgs.lean4
            pkgs.go
          ];
        };
      });

      checks = forAllSystems (system: pkgs:
        let
          vanePkg = self.packages.${system}.vane;
        in
        {
          lean-tests = pkgs.runCommand "vane-lean-tests" { } ''
            ${vanePkg}/bin/vane-test
            touch $out
          '';

          go-tests = pkgs.runCommand "vane-go-tests" {
            nativeBuildInputs = [ pkgs.go ];
          } ''
            cp -r ${self} work
            chmod -R u+w work
            cd work
            export HOME="$TMPDIR"
            export GOCACHE="$TMPDIR/go-cache"
            VANE_BIN=${vanePkg}/bin/vane go test -v ./...
            touch $out
          '';
        });
    };
}
