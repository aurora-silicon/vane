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
              install -Dm755 .lake/build/bin/vane-audit "$out/bin/vane-audit"
              # Compiled .oleans, so vane-audit can inspect them without rebuilding.
              mkdir -p "$out/lib"
              cp -r .lake/build/lib/lean "$out/lib/lean"
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

          goCheck = name: script: pkgs.runCommandCC name {
            nativeBuildInputs = [ pkgs.go ];
          } ''
            cp -r ${self} work
            chmod -R u+w work
            cd work
            export HOME="$TMPDIR"
            export GOCACHE="$TMPDIR/go-cache"
            export GOPATH="$TMPDIR/go"
            export GOPROXY=off
            export GOTOOLCHAIN=local
            ${script}
            touch $out
          '';
        in
        {
          lean-tests = pkgs.runCommand "vane-lean-tests" { } ''
            ${vanePkg}/bin/vane-test
            touch $out
          '';

          # Kernel-level audit of the compiled proofs, then a source scan for
          # proof and trust shortcuts, then full kernel replay with leanchecker.
          proof-audit = pkgs.runCommand "vane-proof-audit" {
            nativeBuildInputs = [ pkgs.python3 ];
          } ''
            export LEAN_SYSROOT=${pkgs.lean4}
            export LEAN_PATH=${vanePkg}/lib/lean:${pkgs.lean4}/lib/lean
            ${vanePkg}/bin/vane-audit
            python3 ${self}/scripts/lean-guard.py forbidden ${self}
            ${pkgs.lean4}/bin/leanchecker Vane Vane.Proofs
            touch $out
          '';

          # The verifier's import closure must stay inside the TCB allowlist.
          tcb-check = pkgs.runCommand "vane-tcb-check" {
            nativeBuildInputs = [ pkgs.python3 ];
          } ''
            python3 ${self}/scripts/lean-guard.py tcb ${self}
            touch $out
          '';

          go-lint = goCheck "vane-go-lint" ''
            unformatted="$(gofmt -l .)"
            if [ -n "$unformatted" ]; then
              echo "gofmt: these files need formatting (run gofmt -w .):" >&2
              echo "$unformatted" >&2
              gofmt -d . >&2
              exit 1
            fi
            if grep -Eq '^[[:space:]]*require([[:space:]]|\(|$)' go.mod; then
              echo "go.mod has a require directive; vane's Go code must have zero external dependencies" >&2
              exit 1
            fi
            if [ -e go.sum ]; then
              echo "go.sum exists; vane's Go code must have zero external dependencies" >&2
              exit 1
            fi
            go vet ./...
            echo "[PASS] go-lint: gofmt clean, go vet clean, zero external dependencies"
          '';

          go-tests = goCheck "vane-go-tests" ''
            CGO_ENABLED=1 VANE_BIN=${vanePkg}/bin/vane go test -race -v ./...
          '';
        });
    };
}
