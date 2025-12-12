plugins {
    `java-library`
    kotlin("jvm") version "2.1.20"
}

group = "com.simiacryptus"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.slf4j:slf4j-api:2.0.16")
    implementation("com.google.code.findbugs:jsr305:3.0.2")
    implementation("com.google.guava:guava:33.1.0-jre")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.15.2")
    implementation("org.json:json:20240303")
    implementation("org.apache.commons:commons-math3:3.6.1")
    implementation("com.google.code.gson:gson:2.10.1")
    implementation("commons-io:commons-io:2.13.0")
    implementation("ch.qos.logback:logback-classic:1.4.11")
    implementation("com.esotericsoftware.kryo:kryo:2.24.0")
    implementation("de.javakaffee:kryo-serializers:0.45")
    implementation("org.apache.commons:commons-lang3:3.13.0")
    implementation("org.jcuda:jcuda:12.6.0") {
        exclude(group = "org.jcuda", module = "jcuda-natives")
    }
    implementation("org.jcuda:jcuda-natives:12.6.0:linux-x86_64")
    implementation("org.jcuda:jcudnn:12.6.0") {
        exclude(group = "org.jcuda", module = "jcudnn-natives")
        exclude(group = "org.jcuda", module = "jcuda-natives")
        exclude(group = "org.jcuda", module = "jcublas-natives")
    }
    implementation("org.jcuda:jcudnn-natives:12.6.0:linux-x86_64")
    implementation("org.jcuda:jcublas-natives:12.6.0:linux-x86_64")
    implementation("org.jblas:jblas:1.2.4")
    implementation("org.eclipse.jgit:org.eclipse.jgit:6.7.0.202309050840-r")
    implementation("com.amazonaws:aws-java-sdk-core:1.12.566")
    implementation("com.amazonaws:aws-java-sdk-s3:1.12.566")
    implementation("com.vladsch.flexmark:flexmark-all:0.64.8")
    implementation("com.vladsch.flexmark:flexmark-pdf-converter:0.64.8")
    testImplementation("guru.nidi:graphviz-java:0.18.1")
    testImplementation("org.apache.commons:commons-compress:1.26.0")
    testImplementation("com.github.haifengl:smile-core:2.6.0")
    testImplementation("com.github.haifengl:smile-plot:2.6.0")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(21)
}