plugins {
    `java-library`
    jacoco
    kotlin("jvm") version "2.1.20"
    id("com.simiacryptus.refcount-autocoder")
}

group = "com.simiacryptus"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    api("com.simiacryptus:refcount-core:2.1.0")
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

tasks.withType<Test> {
    maxHeapSize = "4g"
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    if (System.getProperty("TEST_REPO") == null) {
        systemProperty("TEST_REPO", layout.buildDirectory.dir("reports/mindseye").get().asFile.absolutePath)
    }
    System.getProperty("DEBUG_LIFECYCLE")?.let { systemProperty("DEBUG_LIFECYCLE", it) }
    val cudaLibDirs = listOf("/usr/local/cuda-12.6/lib64", "/usr/local/cuda/lib64")
        .filter { file(it).isDirectory }
    if (cudaLibDirs.isNotEmpty()) {
        val existingLdPath = System.getenv("LD_LIBRARY_PATH").orEmpty()
        val combined = (cudaLibDirs + existingLdPath.split(":").filter { it.isNotBlank() }).distinct().joinToString(":")
        environment("LD_LIBRARY_PATH", combined)
    }
}

tasks.test {
    useJUnitPlatform {
        excludeTags("LayerTest", "Report", "Manual")
    }
    finalizedBy(tasks.jacocoTestReport)
}

tasks.jacocoTestReport {
    dependsOn(tasks.test)
    reports {
        xml.required.set(true)
        html.required.set(true)
    }
}

tasks.register<Test>("layerTests") {
    description = "Runs intensive layer tests."
    group = "verification"
    maxHeapSize = "16g"
    forkEvery = 10
    testClassesDirs = sourceSets.test.get().output.classesDirs
    classpath = sourceSets.test.get().runtimeClasspath
    useJUnitPlatform {
        includeTags("LayerTest")
    }
    extensions.configure<JacocoTaskExtension> {
        isEnabled = false
    }
}

tasks.register<Test>("reportTests") {
    description = "Runs report and dataset tests."
    group = "verification"
    testClassesDirs = sourceSets.test.get().output.classesDirs
    classpath = sourceSets.test.get().runtimeClasspath
    useJUnitPlatform {
        includeTags("Report")
    }
    extensions.configure<JacocoTaskExtension> {
        isEnabled = false
    }
}
java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(22))
    }
}
kotlin {
    jvmToolchain(22)
}