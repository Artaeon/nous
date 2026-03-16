FROM golang:1.22-alpine AS builder

WORKDIR /build
COPY go.mod ./
COPY cmd/ cmd/
COPY internal/ internal/

RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o nous ./cmd/nous

FROM alpine:3.19
RUN apk --no-cache add ca-certificates git

WORKDIR /app
COPY --from=builder /build/nous .
COPY knowledge/ /app/knowledge/

# Memory and workspace volumes
VOLUME ["/data", "/workspace"]

EXPOSE 3333

# Default: server mode with memory in /data
ENTRYPOINT ["./nous"]
CMD ["--serve", "--port", "3333", "--model", "qwen3:4b", "--memory", "/data", "--host", "http://ollama:11434", "--trust"]
