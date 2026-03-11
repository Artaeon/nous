.PHONY: build run clean test test-v test-race test-short ci install serve docker

BINARY := nous
SRC := ./cmd/nous

build:
	go build -o $(BINARY) $(SRC)

run: build
	./$(BINARY)

run-shell: build
	./$(BINARY) --allow-shell

serve: build
	./$(BINARY) --serve --port 3333 --allow-shell --trust

clean:
	rm -f $(BINARY) $(BINARY)-*

test:
	go test ./...

test-v:
	go test ./... -v -count=1

test-race:
	go test ./... -race -count=1

test-short:
	go test ./... -short -count=1

fmt:
	go fmt ./...

vet:
	go vet ./...

lint: fmt vet

ci: fmt vet test test-race

# Install to /opt/nous with systemd service
install: build
	sudo mkdir -p /opt/nous
	sudo cp $(BINARY) /opt/nous/
	sudo cp nous.service /etc/systemd/system/
	sudo systemctl daemon-reload
	@echo "Installed. Start with: sudo systemctl start nous"

# Docker build and run
docker:
	docker compose up --build -d
	@echo "Nous running at http://localhost:3333"

docker-stop:
	docker compose down

# Build for all platforms
release:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o $(BINARY)-linux-amd64 $(SRC)
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -ldflags="-s -w" -o $(BINARY)-linux-arm64 $(SRC)
	CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 go build -ldflags="-s -w" -o $(BINARY)-darwin-amd64 $(SRC)
	CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -ldflags="-s -w" -o $(BINARY)-darwin-arm64 $(SRC)

# Generate Modelfile for custom Ollama model
modelfile: build
	./$(BINARY) --version
	@echo "Run /finetune inside Nous to generate the Modelfile"
