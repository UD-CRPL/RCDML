#!/usr/bin/Rscript

if(require("docopt")){
  print("docopt loaded correctly")
} else {
  print("Trying to install docopt")
  install.packages('docopt', repos="http://cran.r-project.org")
  if (require("docopt")){
    print("docopt installed and loaded")
  } else {
    stop("could not install docopt")
  }
}

if (require("BiocManager")){
  print("Bioconductor loaded correctly")
} else {
  print("Trying to install docopt")
  install.packages("BiocManager", repos="http://cran.r-project.org")
  BiocManager::install(version = "3.15")
  if(require("BiocManager")){
    print("Bioconductor installed and loaded")
  } else {
    stop("could not install Bioconductor")
  }
}


if(require("limma")){
  print("limma loaded correctly")
} else {
  print("Trying to install limma")
  BiocManager::install(c("limma"))
  #source("http://bioconductor.org/biocLite.R")
  #biocLite("limma")
  if (require("limma")){
    print("limma installed and loaded")
  } else {
    stop("could not install limma")
  }
}

if(require("edgeR", quietly = TRUE)){
  print("edgeR loaded correctly")
} else {
  print("Trying to install edgeR")
  BiocManager::install("locfit")
  BiocManager::install(c("edgeR"), dependencies = TRUE, force = TRUE)
  #source("http://bioconductor.org/biocLite.R")
  #biocLite("edgeR")
  if (require("edgeR")){
    print("edgeR installed and loaded")
  } else {
    stop("could not install edgeR")
  }
}

if(require("tidyr")){
  print("tidyr loaded correctly")
} else {
  print("Trying to install tidyr")
  install.packages('tidyr', repos="http://cran.r-project.org")
  if (require("tidyr")){
    print("tidyr installed and loaded")
  } else {
    stop("could not install tidyr")
  }
}

if(require("matrixStats")){
  print("matrixStats loaded correctly")
} else {
  print("Trying to install matrixStats")
  install.packages('matrixStats', repos="http://cran.r-project.org")
  if (require("matrixStats")){
    print("matrixStats installed and loaded")
  } else {
    stop("could not install matrixStats")
  }
}

"
Usage:
          beataml_deg_commandline.r --file=<file> --name=<name> --dir=<dir>
          beataml_deg_commandline.r (-h | --help)

Description:   Runs an edgeR analysis on input RSEM files and conditions.
Options:
          --file=<file>     File detailing samples and groups
          --dir=<dir>       Working directory
          --name=<name>     A prefix for output files
" -> doc

'Runs an edgeR analysis on input RSEM files and conditions.
Usage:
    beataml_deg_commandline.R --file=<file> --name=<name> --dir=<dir>

Options:
    -h --help  Show this screen.
    --file=<file>  File detailing samples and groups
    --dir=<dir>  Working directory
    --name=<name> A prefix for output files

' -> doc

args    <- docopt(doc)

## Args to vars
file <- args$`--file`
name <- args$`--name`
setwd(args$`--dir`)

## Read in our data matrix
#fulltable <- read.delim("read_count_matrix.txt", row.names = 1)
fulltable <- read.delim("genesdf.txt", row.names = 1)
print("data matrix is loaded")

# Load our sample table and subset our data
sampleinfo <- read.delim(file, row.names=1)
sampleinfo <- sampleinfo[ row.names(sampleinfo) %in% colnames(fulltable), ]
cpmtable <- subset(fulltable, select = c(rownames(sampleinfo)))

# Filter out low abundance
thresh <- cpmtable > 1
keep <- rowSums(thresh) >= min(sum(sampleinfo$low), sum(sampleinfo$high))
cpmtable <- cpmtable[keep,]

# Make contrasts
cont_matrix <- makeContrasts(lowVShigh = low-high, levels=as.matrix(sampleinfo))

# Fit the expression matrix to a linear model
fit <- lmFit(cpmtable, as.matrix(sampleinfo))

# Compute contrast
fit_contrast <- contrasts.fit(fit, cont_matrix)

# Bayes statistics of differential expression
fit_contrast <- eBayes(fit_contrast)

# Generate a list of top 200000 differentially expressed genes
# Setting such a high number to get all of them
top_genes <- topTable(fit_contrast, number = 200000, adjust = "BH")
write.table(top_genes, file=paste0(name, "_results.txt"), sep = "\t")
print("list of 20000 differentially expressed genes generated")
